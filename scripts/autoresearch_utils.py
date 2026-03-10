#!/usr/bin/env python3
"""Shared helpers for the FreqAI autoresearch scripts."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import pathlib
from typing import Any, Iterator

try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None  # type: ignore[assignment]


def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_hex_text(text: str) -> str:
    return sha256_hex_bytes(text.encode("utf-8"))


def stable_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def resolve_config_path(freqtrade_dir: pathlib.Path, config_arg: str) -> pathlib.Path:
    path = pathlib.Path(config_arg).expanduser()
    if not path.is_absolute():
        path = (freqtrade_dir / path).resolve()
    return path


def load_config_json(config_path: pathlib.Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return data


def config_fingerprint(config_data: dict[str, Any]) -> str:
    return sha256_hex_text(stable_json_dumps(config_data))


def extract_pair_whitelist(config_data: dict[str, Any]) -> list[str]:
    exchange = config_data.get("exchange")
    if not isinstance(exchange, dict):
        return []
    pair_whitelist = exchange.get("pair_whitelist")
    if not isinstance(pair_whitelist, list):
        return []
    pairs: list[str] = []
    for item in pair_whitelist:
        if isinstance(item, str) and item.strip():
            pairs.append(item.strip())
    return pairs


def profile_fingerprint(profile_data: dict[str, Any]) -> str:
    return sha256_hex_text(stable_json_dumps(profile_data))


def pairlist_fingerprint(pairs: list[str]) -> str:
    normalized = sorted(set(pairs))
    return sha256_hex_text(stable_json_dumps(normalized))


def build_campaign_id(
    strategy: str,
    freqaimodel: str | None,
    train_timerange: str,
    holdout_timerange: str,
    config_fp: str,
    pair_fp: str,
) -> str:
    payload = {
        "strategy": strategy,
        "freqaimodel": freqaimodel or "",
        "train_timerange": train_timerange,
        "holdout_timerange": holdout_timerange,
        "config_fingerprint": config_fp,
        "pairlist_fingerprint": pair_fp,
    }
    digest = sha256_hex_text(stable_json_dumps(payload))
    return digest[:16]


@contextmanager
def exclusive_lock(lock_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """
    Acquire a non-blocking exclusive lock on a lock-file.
    Raises RuntimeError if lock cannot be acquired.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = lock_path.open("a+", encoding="utf-8")
    if fcntl is None:
        fd.close()
        raise RuntimeError("File locking not supported on this platform.")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fd.close()
        raise RuntimeError(f"Another autoresearch process is already running (lock: {lock_path}).") from exc

    try:
        fd.seek(0)
        fd.truncate()
        fd.write(f"pid={os.getpid()}\n")
        fd.flush()
        yield lock_path
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()
