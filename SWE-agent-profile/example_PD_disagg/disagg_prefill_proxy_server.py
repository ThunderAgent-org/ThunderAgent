# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import logging
import os
import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, Response, make_response, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PING_SECONDS = 5

prefill_instances: dict[str, tuple[str, float]] = {}
decode_instances: dict[str, tuple[str, float]] = {}
prefill_cv = threading.Condition()
decode_cv = threading.Condition()
request_counter = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM P/D disaggregation proxy server")
    parser.add_argument("--port", type=int, default=8000, help="Proxy listening port")
    parser.add_argument(
        "--service-host",
        type=str,
        default="0.0.0.0",
        help="Service discovery host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--service-port",
        type=int,
        default=30001,
        help="Service discovery port for prefill/decode registration (default: 30001)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=6 * 60 * 60,
        help="Backend request timeout in seconds",
    )
    parser.add_argument(
        "--prefill-url",
        type=str,
        default="http://localhost:8100/v1/completions",
        help="Unused fallback prefill URL (kept for compatibility)",
    )
    parser.add_argument(
        "--decode-url",
        type=str,
        default="http://localhost:8200/v1/completions",
        help="Unused fallback decode URL (kept for compatibility)",
    )
    return parser.parse_args()


def _remove_expired(instances: dict[str, tuple[str, float]]) -> None:
    now = time.time()
    expired = [key for key, (_, stamp) in instances.items() if stamp <= now]
    for key in expired:
        zmq_addr, _ = instances.pop(key, ("", 0))
        logger.info("🔴 Remove [HTTP:%s, ZMQ:%s]", key, zmq_addr)


def _listen_for_register(poller: zmq.Poller, router_socket: zmq.Socket) -> None:
    while True:
        socks = dict(poller.poll())
        if router_socket not in socks:
            continue
        remote_addr, payload = router_socket.recv_multipart()
        data = msgpack.loads(payload)
        http_addr = data.get("http_address")
        zmq_addr = data.get("zmq_address")
        stamp = time.time() + DEFAULT_PING_SECONDS
        if data.get("type") == "P":
            with prefill_cv:
                prefill_instances[http_addr] = (zmq_addr, stamp)
                _remove_expired(prefill_instances)
                prefill_cv.notify_all()
        elif data.get("type") == "D":
            with decode_cv:
                decode_instances[http_addr] = (zmq_addr, stamp)
                _remove_expired(decode_instances)
                decode_cv.notify_all()
        else:
            logger.warning("Unexpected registration from %s: %s", remote_addr, data)
            continue
        logger.info("🔵 Register [HTTP:%s, ZMQ:%s]", http_addr, zmq_addr)


def start_service_discovery(hostname: str, port: int) -> None:
    if not hostname:
        hostname = socket.gethostname()
    if port <= 0:
        raise ValueError("Service discovery port must be > 0")
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    thread = threading.Thread(
        target=_listen_for_register, args=(poller, router_socket), daemon=True
    )
    thread.start()


def random_uuid() -> str:
    return uuid.uuid4().hex


async def execute_prefill(
    url: str,
    data: dict[str, Any],
    request_id: str,
    timeout: aiohttp.ClientTimeout,
) -> tuple[float, float]:
    headers: dict[str, str] = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers["X-Request-Id"] = request_id
    start_wall = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url=url, json=data, headers=headers) as response:
                if response.status != 200:
                    body = await response.text()
                    logger.error("Prefill backend error %s: %s", response.status, body)
                    raise RuntimeError("Prefill request failed")
                async for _ in response.content.iter_chunked(1024):
                    continue
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            logger.error("Prefill request connection failed: %s", exc)
            raise RuntimeError("Prefill connection failed") from exc
    end_wall = time.time()
    return start_wall, end_wall


def build_decode_stream(
    url: str,
    data: dict[str, Any],
    request_id: str,
    timeout: aiohttp.ClientTimeout,
    *,
    prefill_start: float,
    prefill_http_end: float,
    decode_request_start: float,
) -> Any:
    async def _generator() -> Any:
        headers: dict[str, str] = {}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers["X-Request-Id"] = request_id

        yield f'data: {{"pd_request_id": "{request_id}"}}\n\n'.encode("utf-8")

        first_chunk_wall: float | None = None
        decode_end = decode_request_start
        done_chunk: bytes | None = None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url=url, json=data, headers=headers) as response:
                if response.status != 200:
                    body = await response.text()
                    logger.error("Decode backend error %s: %s", response.status, body)
                    yield body.encode("utf-8")
                    now = time.time()
                    metrics_payload = {
                        "prefill_start": prefill_start,
                        "prefill_http_end": prefill_http_end,
                        "prefill_end": decode_request_start,
                        "decode_request_start": decode_request_start,
                        "decode_start": decode_request_start,
                        "decode_end": now,
                        "prefill_time": max(0.0, decode_request_start - prefill_start),
                        "decode_wait_time": 0.0,
                        "decode_time": 0.0,
                    }
                    yield f"data: {json.dumps({'pd_metrics': metrics_payload})}\n\n".encode("utf-8")
                    return

                async for chunk in response.content.iter_chunked(1024):
                    stripped = chunk.strip()
                    if stripped.endswith(b"[DONE]"):
                        done_chunk = chunk
                        continue
                    yield chunk
                    now = time.time()
                    if first_chunk_wall is None:
                        first_chunk_wall = now
                    decode_end = now

        if first_chunk_wall is None:
            first_chunk_wall = decode_end

        if decode_end <= first_chunk_wall:
            decode_end = time.time()

        decode_start_wall = first_chunk_wall if first_chunk_wall is not None else decode_request_start
        prefill_time = max(0.0, decode_start_wall - prefill_start)
        decode_wait_time = max(0.0, decode_start_wall - decode_request_start)
        decode_time = max(0.0, decode_end - decode_start_wall)

        metrics_payload = {
            "prefill_start": prefill_start,
            "prefill_http_end": prefill_http_end,
            "prefill_end": decode_start_wall,
            "decode_request_start": decode_request_start,
            "decode_start": decode_start_wall,
            "decode_end": decode_end,
            "prefill_time": prefill_time,
            "decode_wait_time": decode_wait_time,
            "decode_time": decode_time,
        }
        yield f"data: {json.dumps({'pd_metrics': metrics_payload})}\n\n".encode("utf-8")
        if done_chunk is not None:
            yield done_chunk

    return _generator()


def _prepare_prefill_payload(payload: dict[str, Any], is_chat: bool) -> dict[str, Any]:
    prefill_payload = dict(payload)
    prefill_payload["stream"] = False
    prefill_payload["max_tokens"] = 1
    prefill_payload.pop("max_output_tokens", None)
    prefill_payload.pop("enable_thinking", None)
    if is_chat:
        prefill_payload["max_completion_tokens"] = 1
    else:
        prefill_payload.pop("max_completion_tokens", None)
    return prefill_payload


def _select_instance(instances: dict[str, tuple[str, float]], cv: threading.Condition) -> list[tuple[str, tuple[str, float]]]:
    with cv:
        start = time.time()
        while not instances:
            remaining = DEFAULT_PING_SECONDS - (time.time() - start)
            if remaining <= 0:
                raise RuntimeError("No instances registered")
            cv.wait(timeout=remaining)
            _remove_expired(instances)
        entries = list(instances.items())
        return entries


def main() -> None:
    args = parse_args()
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    app = Quart(__name__)
    start_service_discovery(args.service_host, args.service_port)

    @app.route("/v1/completions", methods=["POST"])
    @app.route("/v1/chat/completions", methods=["POST"])
    async def handle_request() -> Response:
        path = request.path
        is_chat = path.endswith("/chat/completions")
        try:
            payload = await request.get_json()
            if not isinstance(payload, dict):
                return Response(
                    response=b'{"error":"Request body must be JSON object"}',
                    status=400,
                    content_type="application/json",
                )

            prefill_list = _select_instance(prefill_instances, prefill_cv)
            decode_list = _select_instance(decode_instances, decode_cv)
            global request_counter
            prefill_idx = request_counter % len(prefill_list)
            decode_idx = request_counter % len(decode_list)
            logger.info(
                "dispatch count=%d prefill_idx=%d decode_idx=%d prefill_choices=%s decode_choices=%s",
                request_counter,
                prefill_idx,
                decode_idx,
                [http for http, _ in prefill_list],
                [http for http, _ in decode_list],
            )
            prefill_http, (prefill_zmq, _) = prefill_list[prefill_idx]
            decode_http, (decode_zmq, _) = decode_list[decode_idx]
            request_counter += 1

            # Compose request IDs understood by P2pNcclConnector
            request_id = (
                f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{random_uuid()}"
            )
            prefill_payload = _prepare_prefill_payload(payload, is_chat)

            try:
                prefill_start, prefill_end_http = await execute_prefill(
                    f"http://{prefill_http}{path}", prefill_payload, request_id, timeout
                )
            except RuntimeError as exc:
                logger.error("Prefill request failed for %s: %s", request_id, exc)
                return Response(
                    response=b'{"error":"Prefill backend unavailable"}',
                    status=503,
                    content_type="application/json",
                )

            decode_request_start = time.time()
            decode_stream = build_decode_stream(
                f"http://{decode_http}{path}",
                payload,
                request_id,
                timeout,
                prefill_start=prefill_start,
                prefill_http_end=prefill_end_http,
                decode_request_start=decode_request_start,
            )

            response = await make_response(decode_stream)
            response.timeout = None
            response.headers.setdefault("Content-Type", "text/event-stream")
            response.headers["X-PD-Request-ID"] = request_id
            return response
        except RuntimeError as exc:
            logger.error("No available instances: %s", exc)
            return Response(
                response=b'{"error":"No backend instances registered"}',
                status=503,
                content_type="application/json",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Error processing request")
            return Response(
                response=b'{"error":"Internal server error"}',
                status=500,
                content_type="application/json",
            )

    @app.route("/metrics", methods=["GET"])
    async def metrics() -> Response:
        return Response(response="OK\n", status=200, content_type="text/plain")

    app.run(port=args.port)


if __name__ == "__main__":
    main()
