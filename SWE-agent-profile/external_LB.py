"""Simple external load balancer for vLLM data-parallel deployments.

Launch each ``vllm serve`` data-parallel rank as a standalone process (with
distinct ``--port`` / ``--data-parallel-rank``) and point this script at the
resulting HTTP endpoints. The balancer polls Prometheus metrics exposed by each
rank, chooses a target based on queue depth, and proxies OpenAI-compatible
traffic transparently.

Usage example
-------------

Start each rank::

    CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL \
      --data-parallel-size 2 --data-parallel-rank 0 --port 9000

    CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL \
      --data-parallel-size 2 --data-parallel-rank 1 --port 9001 \
      --data-parallel-address 127.0.0.1 --data-parallel-rpc-port 13345

Run the balancer::

    python external_dp_load_balancer.py \
      --backend http://127.0.0.1:9000 \
      --backend http://127.0.0.1:9001 \
      --listen-host 0.0.0.0 --listen-port 8000

Send OpenAI API requests to ``http://localhost:8000`` instead of individual
rank endpoints. See ``/__lb_state`` for live backend status.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from collections import OrderedDict

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask


logger = logging.getLogger("external_dp_lb")


@dataclass
class Backend:
    name: str
    base_url: str
    metrics_path: str = "/metrics"
    health_path: str = "/health"
    timeout: float = 2.0

    healthy: bool = False
    score: float = float("inf")
    last_updated: float = field(default_factory=lambda: 0.0)
    last_error: str | None = None

    async def update_metrics(self, client: httpx.AsyncClient) -> None:
        metrics_url = self.base_url + self.metrics_path
        start = time.time()
        try:
            resp = await client.get(metrics_url, timeout=self.timeout)
            resp.raise_for_status()
            waiting, running = _parse_load_metrics(resp.text)
            self.score = waiting + running
            self.healthy = True
            self.last_error = None
            self.last_updated = start
            logger.debug(
                "Backend %s metrics: waiting=%s running=%s score=%s",
                self.name,
                waiting,
                running,
                self.score,
            )
        except Exception as exc:  # noqa: BLE001 - reference example
            self.healthy = False
            self.score = float("inf")
            self.last_error = str(exc)
            logger.warning("Backend %s metrics polling failed: %s", self.name, exc)

    async def check_health(self, client: httpx.AsyncClient) -> None:
        health_url = self.base_url + self.health_path
        try:
            resp = await client.get(health_url, timeout=self.timeout)
            self.healthy = resp.status_code == 200
            if not self.healthy:
                self.last_error = f"unexpected status {resp.status_code}"
        except Exception as exc:  # noqa: BLE001 - reference example
            self.healthy = False
            self.last_error = str(exc)


def _parse_load_metrics(text: str) -> Tuple[float, float]:
    waiting = 0.0
    running = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_waiting"):
            waiting = float(line.split()[-1])
        elif line.startswith("vllm:num_requests_running"):
            running = float(line.split()[-1])
    return waiting, running


class BackendPool:
    def __init__(
        self,
        backends: Iterable[Backend],
        poll_interval: float = 1.0,
        affinity_cache_size: int = 10000,
    ):
        self.backends: List[Backend] = list(backends)
        if not self.backends:
            raise ValueError("At least one backend must be provided")
        self.poll_interval = poll_interval
        self._rr_cursor = 0
        self._client = httpx.AsyncClient()
        self._affinity: OrderedDict[str, Backend] = OrderedDict()
        self._affinity_cache_size = affinity_cache_size

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop())

    async def shutdown(self) -> None:
        await self._client.aclose()

    async def _poll_loop(self) -> None:
        while True:
            for backend in self.backends:
                await backend.update_metrics(self._client)
                if not backend.healthy:
                    await backend.check_health(self._client)
            await asyncio.sleep(self.poll_interval)

    def choose_backend(
        self,
        request_id: str | None = None,
        imbalance_threshold: float | None = None,
    ) -> Backend:
        healthy = [b for b in self.backends if b.healthy]
        if not healthy:
            raise HTTPException(status_code=503, detail="No healthy backends")

        fallback_backend = self._select_lowest_score_backend(healthy)
        chosen = fallback_backend

        if request_id and imbalance_threshold is not None:
            preferred = self._affinity.get(request_id)
            if preferred in healthy:
                imbalance = self._compute_imbalance(healthy, preferred)
                if imbalance <= imbalance_threshold:
                    chosen = preferred
                else:
                    logger.debug(
                        "Affinity fallback triggered for %s (imbalance %.4f > %.4f)",
                        request_id,
                        imbalance,
                        imbalance_threshold,
                    )
            else:
                self._affinity.pop(request_id, None)

        logger.debug("Routing request to %s (score=%.2f)", chosen.name, chosen.score)
        return chosen

    def remember_affinity(self, request_id: str, backend: Backend) -> None:
        self._affinity[request_id] = backend
        self._affinity.move_to_end(request_id)
        if len(self._affinity) > self._affinity_cache_size:
            self._affinity.popitem(last=False)

    def forget_affinity(self, request_id: str) -> None:
        self._affinity.pop(request_id, None)

    def _select_lowest_score_backend(self, healthy: List[Backend]) -> Backend:
        best_score = min(b.score for b in healthy)
        candidates = [b for b in healthy if b.score == best_score]
        index = self._rr_cursor % len(candidates)
        self._rr_cursor = (self._rr_cursor + 1) % len(candidates)
        return candidates[index]

    @staticmethod
    def _compute_imbalance(healthy: List[Backend], preferred: Backend) -> float:
        projected_scores = []
        for backend in healthy:
            score = backend.score
            if backend is preferred:
                score += 1.0
            projected_scores.append(score)
        max_score = max(projected_scores)
        min_score = min(projected_scores)
        min_for_ratio = max(min_score, 1.0)
        return (max_score - min_score) / min_for_ratio


def create_app(
    pool: BackendPool,
    affinity_header: str,
    imbalance_threshold: float,
) -> FastAPI:
    app = FastAPI()

    async def dependency() -> BackendPool:
        return pool

    @app.on_event("startup")
    async def _startup() -> None:
        await pool.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await pool.shutdown()

    @app.get("/__lb_state")
    async def lb_state(pool: BackendPool = Depends(dependency)) -> Dict[str, Any]:
        return {
            "backends": [
                {
                    "name": b.name,
                    "base_url": b.base_url,
                    "healthy": b.healthy,
                    "score": b.score,
                    "last_updated": b.last_updated,
                    "last_error": b.last_error,
                }
                for b in pool.backends
            ]
        }

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy(path: str, request: Request, pool: BackendPool = Depends(dependency)) -> Response:
        affinity_id = request.headers.get(affinity_header) or request.query_params.get(
            "request_id"
        )
        backend = pool.choose_backend(
            request_id=affinity_id,
            imbalance_threshold=imbalance_threshold,
        )
        if affinity_id:
            pool.remember_affinity(affinity_id, backend)
        url = f"{backend.base_url}/{path}" if path else backend.base_url
        headers = dict(request.headers)
        headers.pop("host", None)
        payload = await request.body()

        async with httpx.AsyncClient(timeout=backend.timeout) as client:
            try:
                stream = client.stream(
                    request.method,
                    url,
                    headers=headers,
                    content=payload,
                    params=dict(request.query_params),
                )
                response = await stream.__aenter__()
            except Exception as exc:  # noqa: BLE001 - reference example
                raise HTTPException(status_code=502, detail=f"Backend {backend.name} error: {exc}")

            forward_headers = dict(response.headers)
            background = BackgroundTask(stream.__aexit__, None, None, None)

            if response.is_stream_consumed:
                body = await response.aread()
                await stream.__aexit__(None, None, None)
                return Response(content=body, status_code=response.status_code, headers=forward_headers)

            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=forward_headers,
                background=background,
            )

    return app


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple external DP load balancer for vLLM")
    parser.add_argument(
        "--backend",
        action="append",
        required=True,
        help="Backend base URL (e.g. http://127.0.0.1:9000)",
    )
    parser.add_argument("--listen-host", default="127.0.0.1", help="Host/IP to bind the balancer")
    parser.add_argument("--listen-port", type=int, default=8000, help="Port for the balancer")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Metrics polling interval in seconds")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument(
        "--affinity-threshold",
        type=float,
        default=0.5,
        help=(
            "Maximum allowed (max_score - min_score) / min_score when honoring affinity."
            " Set负值以禁用 affinity 优先。"
        ),
    )
    parser.add_argument(
        "--affinity-header",
        default="X-Request-ID",
        help="HTTP header carrying the task-level identifier for sticky routing.",
    )
    parser.add_argument(
        "--affinity-cache-size",
        type=int,
        default=10000,
        help="Maximum number of affinity entries to retain in memory.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    backends = [Backend(name=f"rank-{idx}", base_url=url.rstrip("/")) for idx, url in enumerate(args.backend)]
    pool = BackendPool(
        backends,
        poll_interval=args.poll_interval,
        affinity_cache_size=args.affinity_cache_size,
    )
    imbalance_threshold = args.affinity_threshold if args.affinity_threshold >= 0 else None
    app = create_app(
        pool,
        affinity_header=args.affinity_header,
        imbalance_threshold=imbalance_threshold if imbalance_threshold is not None else float("inf"),
    )

    import uvicorn

    uvicorn.run(app, host=args.listen_host, port=args.listen_port)


if __name__ == "__main__":
    main()

