#!/usr/bin/env python3
"""LLM patch-generation helpers for trading autoresearch loops."""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any
import urllib.error
import urllib.request

try:
    import requests as _requests
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    _requests = None


class _CompatResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http_{self.status_code}: {self.text}")

    def json(self) -> Any:
        return json.loads(self.text or "{}")


class _RequestsCompat:
    @staticmethod
    def get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> _CompatResponse:
        req = urllib.request.Request(url, headers=headers or {}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8")
                return _CompatResponse(int(resp.status), text)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return _CompatResponse(int(exc.code), body)

    @staticmethod
    def post(
        url: str,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> _CompatResponse:
        raw = b""
        if json is not None:
            raw = __import__("json").dumps(json).encode("utf-8")
        req = urllib.request.Request(url, headers=headers or {}, method="POST", data=raw)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8")
                return _CompatResponse(int(resp.status), text)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return _CompatResponse(int(exc.code), body)


requests = _requests if _requests is not None else _RequestsCompat()


DEFAULT_CREDENTIALS_PATH = pathlib.Path("~/.config/autoresearch/openai.json").expanduser()
UPSTREAM_OWNER = "karpathy"
UPSTREAM_REPO = "autoresearch"
UPSTREAM_RAW_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_OWNER}/{UPSTREAM_REPO}"
UPSTREAM_API_BASE = f"https://api.github.com/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}"


def sha256_hex_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_openai_credentials(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"LLM credentials file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid credentials JSON at {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Credentials JSON must be an object: {path}")

    api_key = payload.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"'api_key' is required in credentials file: {path}")

    out: dict[str, str] = {"api_key": api_key.strip()}
    for key in ("base_url", "organization", "project"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()

    if "base_url" not in out:
        out["base_url"] = "https://api.openai.com/v1"
    return out


def _trim_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(64, max_chars - 128)
    return text[:keep] + "\n...[truncated]..."


def _recent_tsv_rows(results_tsv: pathlib.Path, campaign_id: str, limit: int) -> list[dict[str, str]]:
    if limit <= 0 or not results_tsv.exists():
        return []
    rows: list[dict[str, str]] = []
    with results_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("campaign_id") != campaign_id:
                continue
            rows.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "stage": row.get("stage", ""),
                    "status": row.get("status", ""),
                    "suggestion": row.get("suggestion", ""),
                    "decision_reason": row.get("decision_reason", ""),
                    "score": row.get("score", ""),
                    "profit_pct": row.get("profit_pct", ""),
                    "max_drawdown_pct": row.get("max_drawdown_pct", ""),
                    "sharpe": row.get("sharpe", ""),
                    "trades": row.get("trades", ""),
                    "description": row.get("description", ""),
                }
            )
    return rows[-limit:]


def _recent_review_rows(review_log: pathlib.Path, campaign_id: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not review_log.exists():
        return []
    rows: list[dict[str, Any]] = []
    with review_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("campaign_id") != campaign_id:
                continue
            rows.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "final_suggestion": row.get("final_suggestion"),
                    "description": row.get("description"),
                    "llm": row.get("llm"),
                    "train": row.get("train"),
                    "holdout": row.get("holdout"),
                }
            )
    return rows[-limit:]


def fetch_upstream_context_bundle(cache_dir: pathlib.Path, ref: str, timeout_sec: int) -> dict[str, Any]:
    """
    Fetch optional upstream context snippets from karpathy/autoresearch.
    Returns a dict that is safe to embed into review logs and prompts.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = cache_dir / f"upstream_{ref}.json"

    headers = {"Accept": "application/vnd.github+json", "User-Agent": "autoresearch-llm-patch-engine"}
    commit_sha: str | None = None
    snippets: dict[str, str] = {}
    error: str | None = None

    try:
        commit_resp = requests.get(
            f"{UPSTREAM_API_BASE}/commits/{ref}",
            headers=headers,
            timeout=timeout_sec,
        )
        commit_resp.raise_for_status()
        commit_json = commit_resp.json()
        if isinstance(commit_json, dict):
            sha = commit_json.get("sha")
            if isinstance(sha, str) and sha:
                commit_sha = sha
    except Exception as exc:  # pragma: no cover - network variability
        error = f"commit_fetch_failed: {exc}"

    for rel in ("README.md", "program.md", "program.trading.md", "train.py"):
        url = f"{UPSTREAM_RAW_BASE}/{ref}/{rel}"
        try:
            resp = requests.get(url, timeout=timeout_sec)
            resp.raise_for_status()
            snippets[rel] = _trim_text(resp.text, 8000 if rel != "train.py" else 12000)
        except Exception as exc:  # pragma: no cover - network variability
            if error is None:
                error = f"snippet_fetch_failed: {exc}"

    payload = {
        "ref": ref,
        "commit": commit_sha,
        "snippets": snippets,
        "error": error,
    }
    bundle_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                    if parts:
                        return "\n".join(parts)
    return ""


def extract_unified_diff(text: str) -> str:
    raw = text.strip()
    if not raw:
        return ""

    fence = re.match(r"^```(?:diff|patch)?\n(?P<body>[\s\S]*?)\n```$", raw, flags=re.IGNORECASE)
    if fence:
        raw = fence.group("body").strip()

    if raw.startswith("diff --git") or raw.startswith("--- ") or raw.startswith("*** "):
        return raw

    idx = raw.find("diff --git")
    if idx >= 0:
        return raw[idx:].strip()
    idx = raw.find("--- ")
    if idx >= 0:
        return raw[idx:].strip()
    return ""


def _normalize_patch_path(token: str) -> str:
    token = token.strip().split("\t", 1)[0].strip()
    if token.startswith("a/") or token.startswith("b/"):
        token = token[2:]
    return token


def validate_patch_targets(patch_text: str, allowed_relative_path: str) -> tuple[bool, str | None, list[str]]:
    touched: set[str] = set()
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            token = line[4:].strip()
            if token != "/dev/null":
                touched.add(_normalize_patch_path(token))
        elif line.startswith("+++ "):
            token = line[4:].strip()
            if token != "/dev/null":
                touched.add(_normalize_patch_path(token))

    if not touched:
        return False, "patch_has_no_file_targets", []

    touched_list = sorted(touched)
    if len(touched_list) != 1:
        return False, f"patch_touches_multiple_files:{','.join(touched_list)}", touched_list
    if touched_list[0] != allowed_relative_path:
        return False, f"patch_touches_disallowed_file:{touched_list[0]}", touched_list
    return True, None, touched_list


def apply_unified_diff(repo_dir: pathlib.Path, patch_text: str) -> tuple[bool, str | None]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)
        tmp.write(patch_text)

    try:
        check = subprocess.run(
            ["git", "apply", "--check", "--recount", str(tmp_path)],
            cwd=repo_dir,
            text=True,
            capture_output=True,
        )
        if check.returncode != 0:
            err = (check.stdout or "") + (check.stderr or "")
            return False, f"git_apply_check_failed: {err.strip()}"

        apply_proc = subprocess.run(
            ["git", "apply", "--recount", str(tmp_path)],
            cwd=repo_dir,
            text=True,
            capture_output=True,
        )
        if apply_proc.returncode != 0:
            err = (apply_proc.stdout or "") + (apply_proc.stderr or "")
            return False, f"git_apply_failed: {err.strip()}"
        return True, None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def run_py_compile(target_file: pathlib.Path) -> tuple[bool, str | None]:
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(target_file)],
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return True, None
    out = (proc.stdout or "") + (proc.stderr or "")
    return False, f"py_compile_failed: {out.strip()}"


def build_prompt_payload(
    *,
    target_relative_path: str,
    target_source: str,
    recent_results: list[dict[str, str]],
    recent_review: list[dict[str, Any]],
    upstream_bundle: dict[str, Any] | None,
    repair_hint: str | None,
    previous_patch: str | None,
) -> tuple[str, str, str]:
    system_prompt = (
        "You are an expert quantitative ML engineer improving a FreqAI LSTM regressor.\n"
        "Output ONLY a unified diff patch.\n"
        f"You may modify ONLY: {target_relative_path}\n"
        "Do not include explanations. Do not include markdown fences."
    )

    upstream_text = "{}"
    if upstream_bundle:
        snippets = upstream_bundle.get("snippets", {})
        if isinstance(snippets, dict):
            upstream_text = json.dumps(
                {
                    "ref": upstream_bundle.get("ref"),
                    "commit": upstream_bundle.get("commit"),
                    "error": upstream_bundle.get("error"),
                    "snippets": snippets,
                },
                indent=2,
                sort_keys=True,
            )

    user_prompt = (
        f"TARGET FILE: {target_relative_path}\n"
        "GOAL: improve train+holdout trading outcomes while preserving code correctness.\n"
        "STRICT CONSTRAINTS:\n"
        "- Single-file patch only\n"
        "- Keep API compatibility for FreqAI model class\n"
        "- Do not touch other files\n\n"
        "CURRENT FILE:\n"
        f"{target_source}\n\n"
        "RECENT RESULTS (last rows):\n"
        f"{json.dumps(recent_results, indent=2, sort_keys=True)}\n\n"
        "RECENT REVIEW (last rows):\n"
        f"{json.dumps(recent_review, indent=2, sort_keys=True)}\n\n"
        "UPSTREAM CONTEXT (optional):\n"
        f"{upstream_text}\n\n"
    )
    if repair_hint:
        user_prompt += f"PREVIOUS ATTEMPT ERROR:\n{repair_hint}\n\n"
    if previous_patch:
        user_prompt += f"PREVIOUS PATCH:\n{previous_patch}\n\n"

    user_prompt += "Now provide the next unified diff patch."
    prompt_hash = sha256_hex_text(system_prompt + "\n" + user_prompt)
    return system_prompt, user_prompt, prompt_hash


def generate_patch_with_openai(
    *,
    credentials_file: pathlib.Path,
    model: str,
    target_relative_path: str,
    target_source: str,
    results_tsv: pathlib.Path,
    review_log: pathlib.Path,
    campaign_id: str,
    context_rows: int,
    timeout_sec: int,
    upstream_bundle: dict[str, Any] | None = None,
    repair_hint: str | None = None,
    previous_patch: str | None = None,
) -> dict[str, Any]:
    creds = load_openai_credentials(credentials_file)
    base_url = creds["base_url"].rstrip("/")
    endpoint = f"{base_url}/responses"

    recent_results = _recent_tsv_rows(results_tsv, campaign_id, context_rows)
    recent_review = _recent_review_rows(review_log, campaign_id, context_rows)
    system_prompt, user_prompt, prompt_hash = build_prompt_payload(
        target_relative_path=target_relative_path,
        target_source=target_source,
        recent_results=recent_results,
        recent_review=recent_review,
        upstream_bundle=upstream_bundle,
        repair_hint=repair_hint,
        previous_patch=previous_patch,
    )

    headers = {
        "Authorization": f"Bearer {creds['api_key']}",
        "Content-Type": "application/json",
    }
    if "organization" in creds:
        headers["OpenAI-Organization"] = creds["organization"]
    if "project" in creds:
        headers["OpenAI-Project"] = creds["project"]

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("OpenAI response payload is not a JSON object.")

    raw_text = _extract_response_text(data)
    patch_text = extract_unified_diff(raw_text)
    usage = data.get("usage")
    if usage is not None and not isinstance(usage, dict):
        usage = None
    request_id = data.get("id")
    if not isinstance(request_id, str):
        request_id = None

    return {
        "request_id": request_id,
        "prompt_hash": prompt_hash,
        "patch_text": patch_text,
        "raw_text": raw_text,
        "usage": usage,
    }
