"""Slime adapter router definitions."""

import logging
from typing import Any, Callable, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ...scheduler import MultiBackendRouter
from .utils import (
    extract_sglang_usage,
    extract_worker_url,
    format_workers,
    read_json,
    resolve_worker_url,
)


def create_slime_adapter(
    state_router: MultiBackendRouter,
    program_id_getter: Callable[[Dict[str, Any]], str],
    logger: logging.Logger,
) -> APIRouter:
    """Create Slime-compatible adapter router."""
    adapter_router = APIRouter()

    _register_generate_route(
        adapter_router=adapter_router,
        state_router=state_router,
        program_id_getter=program_id_getter,
        logger=logger,
    )
    _register_worker_routes(
        adapter_router=adapter_router,
        state_router=state_router,
    )
    return adapter_router


# -------------------------------------------------------------------------
# Route Registration
# -------------------------------------------------------------------------

def _register_generate_route(
    *,
    adapter_router: APIRouter,
    state_router: MultiBackendRouter,
    program_id_getter: Callable[[Dict[str, Any]], str],
    logger: logging.Logger,
) -> None:
    """Register Slime-compatible /generate route."""

    @adapter_router.post("/generate")
    async def generate(request: Request):
        """Handle Slime/SGLang-style generate requests with TR scheduling semantics."""
        payload = await read_json(request)

        if not state_router.backends:
            raise HTTPException(status_code=503, detail="No workers available. Add workers via /add_worker first.")

        program_id = program_id_getter(payload)
        program_state = state_router.get_or_create_program(program_id)

        if program_state.profile:
            program_state.profile.on_request_arrive()

        await state_router.update_program_before_request(program_id, program_state, payload)

        if program_state.profile:
            program_state.profile.on_request_start()

        backend = state_router.get_backend_for_program(program_id)
        try:
            output = await state_router.proxy_json_post(backend, "/generate", payload)
        except Exception as exc:
            logger.exception("Failed to proxy /generate request to %s", backend.url)
            raise HTTPException(status_code=502, detail=f"Backend generate request failed: {exc}") from exc

        total_tokens, prompt_tokens, cached_tokens = extract_sglang_usage(output)
        state_router.update_program_after_request(program_id, program_state, total_tokens, prompt_tokens)
        if program_state.profile:
            program_state.profile.on_request_end(prompt_tokens, cached_tokens)

        return JSONResponse(output)


def _register_worker_routes(
    *,
    adapter_router: APIRouter,
    state_router: MultiBackendRouter,
) -> None:
    """Register worker management routes used by Slime/SGLang router clients."""

    @adapter_router.post("/add_worker")
    async def add_worker(request: Request):
        """SGLang-router compatible endpoint to add worker."""
        worker_url = await extract_worker_url(request)
        added = await state_router.add_backend(worker_url)
        return JSONResponse(
            {"status": "success", "added": added, "worker_urls": state_router.list_backend_urls()}
        )

    @adapter_router.post("/remove_worker")
    async def remove_worker(request: Request):
        """SGLang-router compatible endpoint to remove worker."""
        worker_url = await extract_worker_url(request)
        try:
            removed = await state_router.remove_backend(worker_url)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        if not removed:
            raise HTTPException(status_code=404, detail=f"Worker not found: {worker_url}")

        return JSONResponse(
            {"status": "success", "removed": True, "worker_urls": state_router.list_backend_urls()}
        )

    @adapter_router.get("/list_workers")
    async def list_workers():
        """SGLang-router compatible endpoint to list workers."""
        return JSONResponse({"urls": state_router.list_backend_urls()})

    @adapter_router.get("/workers")
    async def workers():
        """Newer router-compatible worker listing endpoint."""
        return JSONResponse({"workers": format_workers(state_router.list_backend_urls())})

    @adapter_router.post("/workers")
    async def add_worker_v2(request: Request):
        """Newer router-compatible worker add endpoint."""
        worker_url = await extract_worker_url(request)
        added = await state_router.add_backend(worker_url)
        return JSONResponse(
            {"status": "success", "added": added, "workers": format_workers(state_router.list_backend_urls())}
        )

    @adapter_router.delete("/workers/{worker_ref}")
    async def remove_worker_v2(worker_ref: str):
        """Newer router-compatible worker remove endpoint by id or URL."""
        workers = format_workers(state_router.list_backend_urls())
        worker_url = resolve_worker_url(worker_ref, workers)

        try:
            removed = await state_router.remove_backend(worker_url)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        if not removed:
            raise HTTPException(status_code=404, detail=f"Worker not found: {worker_url}")

        return JSONResponse(
            {"status": "success", "removed": True, "workers": format_workers(state_router.list_backend_urls())}
        )
