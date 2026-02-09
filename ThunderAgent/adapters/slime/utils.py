"""Utility helpers for Slime adapter routes."""

from typing import Any, Dict, List
from urllib.parse import unquote

from fastapi import HTTPException, Request


def to_int(value: Any, default: int = 0) -> int:
    """Safely cast a value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def read_json(request: Request) -> Dict[str, Any]:
    """Read JSON object from request, returning 400 on invalid payload."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return payload


async def read_json_silent(request: Request) -> Dict[str, Any]:
    """Best-effort JSON read, returning empty object when payload is unavailable."""
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_sglang_usage(output: Dict[str, Any]) -> tuple[int, int, int]:
    """Extract (total_tokens, prompt_tokens, cached_tokens) from SGLang response."""
    meta_info = output.get("meta_info", {})
    if not isinstance(meta_info, dict):
        meta_info = {}

    prompt_tokens = to_int(meta_info.get("prompt_tokens"), default=0)
    cached_tokens = to_int(meta_info.get("cached_tokens"), default=0)
    completion_tokens = to_int(meta_info.get("completion_tokens"), default=0)
    if completion_tokens <= 0:
        output_token_logprobs = meta_info.get("output_token_logprobs")
        if isinstance(output_token_logprobs, list):
            completion_tokens = len(output_token_logprobs)

    total_tokens = to_int(meta_info.get("total_tokens"), default=prompt_tokens + completion_tokens)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    return total_tokens, prompt_tokens, cached_tokens


async def extract_worker_url(request: Request) -> str:
    """Extract worker URL from query string or JSON body."""
    worker_url = request.query_params.get("url") or request.query_params.get("worker_url")
    if worker_url:
        return worker_url.rstrip("/")

    payload = await read_json_silent(request)
    worker_url = payload.get("url") or payload.get("worker_url")
    if worker_url:
        return str(worker_url).rstrip("/")

    raise HTTPException(status_code=400, detail="worker_url is required (query ?url=... or JSON body).")


def format_workers(worker_urls: List[str]) -> List[Dict[str, Any]]:
    """Convert backend URLs to router-compatible worker objects."""
    return [{"id": idx, "url": url} for idx, url in enumerate(worker_urls)]


def resolve_worker_url(worker_ref: str, workers: List[Dict[str, Any]]) -> str:
    """Resolve worker reference from numeric id or URL path component."""
    if worker_ref.isdigit():
        worker_id = int(worker_ref)
        if worker_id < 0 or worker_id >= len(workers):
            raise HTTPException(status_code=404, detail=f"Worker id not found: {worker_id}")
        return workers[worker_id]["url"]
    return unquote(worker_ref).rstrip("/")
