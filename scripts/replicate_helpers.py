"""Shared Replicate plumbing for the Cast a Shadow generation scripts.

Wraps `replicate.Client` with utilities that:
  - Discover input schemas at runtime (so we can adapt to model-version drift)
  - Submit single-shot generations with no retries (paid POSTs are not idempotent)
  - Download outputs to a target Path
  - Write structured manifests next to every artifact

Used by every generate_*.py and the stitcher orchestration. Designed to be
small, vendor-agnostic at the call site, and easy to adapt when Replicate
adds or removes fields on a model.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


# --- Rate limiting ---------------------------------------------------------
# Replicate enforces 6 prediction-creates / minute with burst=1 on accounts
# below $5 credit. The submit lock guarantees we wait at least N seconds
# between client.run() calls, regardless of worker count.

_SUBMIT_LOCK = threading.Lock()
_LAST_SUBMIT_TIME = 0.0
_SUBMIT_INTERVAL = 12.0


def wait_for_submit_slot() -> None:
    """Block until at least _SUBMIT_INTERVAL has passed since the last submit."""
    global _LAST_SUBMIT_TIME
    with _SUBMIT_LOCK:
        elapsed = time.time() - _LAST_SUBMIT_TIME
        if elapsed < _SUBMIT_INTERVAL:
            time.sleep(_SUBMIT_INTERVAL - elapsed)
        _LAST_SUBMIT_TIME = time.time()


# --- Client construction ---------------------------------------------------


def load_replicate_client() -> Any:
    """Return a Replicate client built from REPLICATE_API_TOKEN."""
    import os

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise SystemExit("REPLICATE_API_TOKEN is required.")
    import replicate

    return replicate.Client(api_token=token)


# --- Schema introspection --------------------------------------------------


def split_slug(slug: str) -> tuple[str, str] | None:
    parts = slug.split("/")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def schema_fields(client: Any, slug: str, logger: logging.Logger) -> set[str]:
    """Return the set of valid input field names for a Replicate model.

    Strips an optional ":version" suffix before lookup so version-pinned slugs
    work the same as bare slugs. Returns empty set on any error so callers
    can fall back to sending all candidate fields.
    """
    try:
        bare_slug = slug.split(":", 1)[0]
        owner_name = split_slug(bare_slug)
        if not owner_name:
            return set()
        try:
            model = client.models.get(bare_slug)
        except TypeError:
            model = client.models.get(*owner_name)
        version = getattr(model, "latest_version", None)
        schema = (
            getattr(version, "openapi_schema", None)
            or getattr(model, "openapi_schema", None)
            or {}
        )
        components = schema.get("components", {}).get("schemas", {})
        input_schema = components.get("Input") or components.get("InputSchema") or {}
        props = input_schema.get("properties", {})
        if props:
            fields = set(props)
            logger.info("schema fields for %s: %s", slug, ", ".join(sorted(fields)))
            return fields
    except Exception as exc:  # noqa: BLE001 - schema lookup never blocks generation.
        logger.info("schema lookup skipped for %s: %s", slug, str(exc)[:180])
    return set()


def keep_supported_fields(payload: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    """Drop None values, then drop keys not in `fields` (if `fields` is set)."""
    if not fields:
        return {k: v for k, v in payload.items() if v is not None}
    return {k: v for k, v in payload.items() if k in fields and v is not None}


# --- Payload prep ----------------------------------------------------------


def public_payload(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    """Stringify Path objects relative to repo root for manifest logging."""

    def clean(value: Any) -> Any:
        if isinstance(value, Path):
            try:
                return str(value.relative_to(root))
            except ValueError:
                return str(value)
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value

    return {k: clean(v) for k, v in payload.items()}


def open_file_inputs(value: Any, handles: list[Any]) -> Any:
    """Replace any Path values with open file handles. Caller must close handles."""
    if isinstance(value, Path):
        handle = value.open("rb")
        handles.append(handle)
        return handle
    if isinstance(value, list):
        return [open_file_inputs(v, handles) for v in value]
    if isinstance(value, dict):
        return {k: open_file_inputs(v, handles) for k, v in value.items()}
    return value


# --- Output handling -------------------------------------------------------


def first_url(output: Any) -> str | None:
    """Pull the first downloadable URL out of any Replicate output shape."""
    if isinstance(output, str) and output.startswith(("http://", "https://")):
        return output
    if isinstance(output, list):
        for item in output:
            url = first_url(item)
            if url:
                return url
    if isinstance(output, dict):
        for key in ("url", "video_url", "image_url", "output"):
            value = output.get(key)
            url = first_url(value)
            if url:
                return url
        for key in ("video", "image", "images", "files"):
            value = output.get(key)
            url = first_url(value)
            if url:
                return url
    url = getattr(output, "url", None)
    if isinstance(url, str):
        return url
    text = str(output)
    if text.startswith(("http://", "https://")):
        return text
    return None


def summarize_output(output: Any) -> Any:
    if isinstance(output, (str, int, float, bool)) or output is None:
        return output
    if isinstance(output, list):
        return [summarize_output(v) for v in output[:3]]
    if isinstance(output, dict):
        return {k: summarize_output(v) for k, v in list(output.items())[:8]}
    url = getattr(output, "url", None)
    if url:
        return {"url": url}
    return str(output)[:300]


def write_output(output: Any, path: Path) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(output, "read"):
        data = output.read()
        path.write_bytes(data)
        return None
    url = first_url(output)
    if not url:
        raise RuntimeError(f"No downloadable URL in Replicate output: {summarize_output(output)}")
    req = urllib.request.Request(url, headers={"User-Agent": "cast-a-shadow"})
    with urllib.request.urlopen(req, timeout=240) as response:
        path.write_bytes(response.read())
    return url


# --- Generation entry point ------------------------------------------------


def run_replicate(
    client: Any,
    model_slug: str,
    payload: dict[str, Any],
    out_path: Path,
    logger: logging.Logger,
    dry_run: bool,
    repo_root: Path,
    rate_limit: bool = True,
) -> dict[str, Any]:
    """Submit a single Replicate prediction. No retries on paid POST failures.

    A 5xx or network blip mid-creation can produce a duplicate billable job
    on retry. Fail fast and inspect the Replicate dashboard before rerunning.
    """
    clean_payload = public_payload(payload, repo_root)
    if dry_run:
        logger.info("DRY RUN %s -> %s", model_slug, out_path.relative_to(repo_root))
        return {
            "status": "dry_run",
            "model": model_slug,
            "output_path": str(out_path.relative_to(repo_root)),
            "input": clean_payload,
        }

    if rate_limit:
        wait_for_submit_slot()

    handles: list[Any] = []
    try:
        api_payload = open_file_inputs(payload, handles)
        t0 = time.time()
        logger.info("RUN %s -> %s", model_slug, out_path.relative_to(repo_root))
        output = client.run(model_slug, input=api_payload)
        elapsed = time.time() - t0
        output_url = write_output(output, out_path)
        size = out_path.stat().st_size if out_path.exists() else 0
        logger.info("OK %s %.1fs %.1fMB", out_path.name, elapsed, size / 1_000_000)
        return {
            "status": "ok",
            "model": model_slug,
            "output_path": str(out_path.relative_to(repo_root)),
            "output_url": output_url,
            "wall_seconds": round(elapsed, 1),
            "size_bytes": size,
            "input": clean_payload,
            "raw_output": summarize_output(output),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("FAIL %s: %s", out_path.name, exc)
        return {
            "status": "error",
            "model": model_slug,
            "output_path": str(out_path.relative_to(repo_root)),
            "error": str(exc)[:800],
            "input": clean_payload,
        }
    finally:
        for handle in handles:
            handle.close()


def cached_result(path: Path, prompt: str, model: str, repo_root: Path) -> dict[str, Any]:
    return {
        "status": "cached",
        "model": model,
        "output_path": str(path.relative_to(repo_root)),
        "size_bytes": path.stat().st_size,
        "input": {"prompt": prompt},
    }


def overall_status(results: list[dict[str, Any]]) -> str:
    if not results:
        return "empty"
    if all(r.get("status") == "dry_run" for r in results):
        return "dry_run"
    if all(r.get("status") in {"ok", "cached"} for r in results):
        return "ok"
    return "partial"


def setup_logging(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
