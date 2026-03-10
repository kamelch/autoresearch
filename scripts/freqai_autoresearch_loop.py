#!/usr/bin/env python3
"""Autoresearch loop for FreqAI strategy tuning (campaign-isolated v2)."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import random
import re
import shlex
import signal
import subprocess
import shutil
import sys
from typing import Any

from autoresearch_utils import exclusive_lock, profile_fingerprint
import llm_patch_engine


NUM_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        txt = value.strip().replace("%", "")
        if not txt:
            return None
        try:
            return float(txt)
        except ValueError:
            return None
    return None


def decimals_for_step(step: float) -> int:
    txt = f"{step:.12f}".rstrip("0").rstrip(".")
    if "." not in txt:
        return 0
    return len(txt.split(".", 1)[1])


def quantize(value: float, step: float, decimals: int) -> float:
    return round(round(value / step) * step, decimals)


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sanitize_range(lo: float, hi: float, hard_min: float, hard_max: float, step: float) -> tuple[float, float]:
    decimals = decimals_for_step(step)
    lo = quantize(clip(lo, hard_min, hard_max), step, decimals)
    hi = quantize(clip(hi, hard_min, hard_max), step, decimals)
    if hi <= lo:
        if lo + step <= hard_max:
            hi = quantize(lo + step, step, decimals)
        elif hi - step >= hard_min:
            lo = quantize(hi - step, step, decimals)
        else:
            lo = quantize(hard_min, step, decimals)
            hi = quantize(hard_max, step, decimals)
    return lo, hi


def normalize_tunable_spec(name: str, raw: dict[str, Any]) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"Tunable '{name}' must be an object.")
    if "min" not in raw or "max" not in raw or "step" not in raw:
        raise ValueError(f"Tunable '{name}' must contain min/max/step.")

    step = float(raw["step"])
    if step <= 0:
        raise ValueError(f"Tunable '{name}' has non-positive step: {step}")

    cur_min = float(raw["min"])
    cur_max = float(raw["max"])
    if cur_max <= cur_min:
        raise ValueError(f"Tunable '{name}' has invalid range min={cur_min}, max={cur_max}")

    hard_min = float(raw.get("hard_min", cur_min))
    hard_max = float(raw.get("hard_max", cur_max))
    hard_min = min(hard_min, cur_min)
    hard_max = max(hard_max, cur_max)
    if hard_max <= hard_min:
        raise ValueError(f"Tunable '{name}' has invalid hard bounds {hard_min}..{hard_max}")

    cur_min, cur_max = sanitize_range(cur_min, cur_max, hard_min, hard_max, step)
    return {
        "min": cur_min,
        "max": cur_max,
        "step": step,
        "hard_min": hard_min,
        "hard_max": hard_max,
    }


def load_profile(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "strategy_file" not in data or "tunables" not in data:
        raise ValueError("Profile must contain 'strategy_file' and 'tunables'.")
    if not isinstance(data["tunables"], dict) or not data["tunables"]:
        raise ValueError("'tunables' must be a non-empty object.")

    normalized: dict[str, dict[str, float]] = {}
    for name, spec in data["tunables"].items():
        normalized[name] = normalize_tunable_spec(name, spec)

    data["_normalized_tunables"] = normalized
    return data


def read_strategy_text(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")
    return path.read_text(encoding="utf-8")


def parse_value(text: str, name: str) -> float:
    pat = re.compile(rf"(?m)^(?P<prefix>\s*{re.escape(name)}\s*=\s*)(?P<val>{NUM_RE})(?P<suffix>\s*(?:#.*)?)$")
    m = pat.search(text)
    if not m:
        raise ValueError(f"Could not locate numeric assignment for '{name}' in strategy file.")
    return float(m.group("val"))


def mutate_value(current: float, spec: dict[str, Any], rng: random.Random) -> float:
    lo = float(spec["min"])
    hi = float(spec["max"])
    step = float(spec["step"])
    decimals = decimals_for_step(step)

    jump = rng.choice([-3, -2, -1, 1, 2, 3]) * step
    candidate = clip(current + jump, lo, hi)
    candidate = quantize(candidate, step, decimals)

    if candidate == current:
        span = int(round((hi - lo) / step))
        idx = rng.randint(0, max(span, 0))
        candidate = quantize(lo + idx * step, step, decimals)
    return candidate


def weighted_choice(weights: dict[str, float], rng: random.Random) -> str:
    keys = list(weights.keys())
    vals = [max(1e-9, float(weights[k])) for k in keys]
    total = sum(vals)
    pick = rng.random() * total
    acc = 0.0
    for key, w in zip(keys, vals):
        acc += w
        if pick <= acc:
            return key
    return keys[-1]


def replace_assignment(text: str, name: str, new_value: float) -> str:
    pat = re.compile(rf"(?m)^(?P<prefix>\s*{re.escape(name)}\s*=\s*)(?P<val>{NUM_RE})(?P<suffix>\s*(?:#.*)?)$")
    m = pat.search(text)
    if not m:
        raise ValueError(f"Could not replace '{name}', assignment not found.")
    old_token = m.group("val")
    decimals = 0
    if "." in old_token:
        decimals = len(old_token.split(".", 1)[1].rstrip("0"))
    if "e" in old_token.lower():
        val_token = str(new_value)
    elif decimals > 0:
        val_token = f"{new_value:.{decimals}f}"
    else:
        val_token = str(int(round(new_value)))
    return text[: m.start()] + f"{m.group('prefix')}{val_token}{m.group('suffix')}" + text[m.end() :]


def enforce_consistency(text: str) -> str:
    lp = parse_value(text, "long_prob_min")
    sp = parse_value(text, "short_prob_max")
    le = parse_value(text, "long_exit_prob_max")
    se = parse_value(text, "short_exit_prob_min")

    if sp >= lp:
        sp = max(0.0, lp - 0.01)
        text = replace_assignment(text, "short_prob_max", sp)

    if le >= lp:
        le = max(0.0, lp - 0.01)
        text = replace_assignment(text, "long_exit_prob_max", le)

    if se <= sp:
        se = min(1.0, sp + 0.01)
        text = replace_assignment(text, "short_exit_prob_min", se)

    return text


def run_backtest_runner(
    repo_dir: pathlib.Path,
    freqtrade_dir: pathlib.Path,
    config: str,
    freqtrade_bin: str | None,
    strategy: str,
    strategy_path: str | None,
    freqaimodel: str | None,
    freqaimodel_path: str | None,
    timerange: str,
    description: str,
    results_tsv: pathlib.Path,
    dd_penalty: float,
    min_improvement: float,
    campaign_id: str,
    candidate_id: str,
    stage: str,
    min_profit_pct: float,
    max_drawdown_pct: float,
    min_sharpe: float,
    pair_min_trades_floor: int,
    pair_min_trades_mode: str,
) -> subprocess.CompletedProcess[str]:
    runner = repo_dir / "scripts" / "run_freqtrade_backtest.py"
    cmd = [
        sys.executable,
        str(runner),
        "--freqtrade-dir",
        str(freqtrade_dir),
        "--config",
        config,
        "--strategy",
        strategy,
        "--timerange",
        timerange,
        "--repo-dir",
        str(repo_dir),
        "--results-tsv",
        str(results_tsv),
        "--description",
        description,
        "--dd-penalty",
        str(dd_penalty),
        "--min-improvement",
        str(min_improvement),
        "--campaign-id",
        campaign_id,
        "--candidate-id",
        candidate_id,
        "--stage",
        stage,
        "--min-profit-pct",
        str(min_profit_pct),
        "--max-drawdown-pct",
        str(max_drawdown_pct),
        "--min-sharpe",
        str(min_sharpe),
        "--pair-min-trades-floor",
        str(pair_min_trades_floor),
        "--pair-min-trades-mode",
        pair_min_trades_mode,
    ]
    if strategy_path:
        cmd.extend(["--strategy-path", strategy_path])
    if freqaimodel:
        cmd.extend(["--freqaimodel", freqaimodel])
    if freqaimodel_path:
        cmd.extend(["--freqaimodel-path", freqaimodel_path])
    if freqtrade_bin:
        cmd.extend(["--freqtrade-bin", freqtrade_bin])

    return subprocess.run(cmd, text=True, cwd=repo_dir)


def _atomic_write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def init_space_state(
    tunables: dict[str, dict[str, float]],
    profile_raw: dict[str, Any],
    campaign_id: str,
    profile_path: pathlib.Path,
    profile_hash: str,
) -> dict[str, Any]:
    params: dict[str, dict[str, float | int]] = {}
    for name, spec in tunables.items():
        raw_spec = profile_raw["tunables"].get(name, {})
        params[name] = {
            "hard_min": float(spec["hard_min"]),
            "hard_max": float(spec["hard_max"]),
            "active_min": float(spec["min"]),
            "active_max": float(spec["max"]),
            "step": float(spec["step"]),
            "weight": float(raw_spec.get("weight", 1.0)),
            "keep_count": 0,
            "discard_count": 0,
            "crash_count": 0,
        }
    return {
        "version": 2,
        "campaign_id": campaign_id,
        "profile_hash": profile_hash,
        "profile_path": str(profile_path),
        "params": params,
    }


def _merge_loaded_params(
    state: dict[str, Any],
    loaded: dict[str, Any],
    tunables: dict[str, dict[str, float]],
    profile_raw: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(loaded, dict) or not isinstance(loaded.get("params"), dict):
        return state

    for name, spec in tunables.items():
        item = loaded["params"].get(name)
        if not isinstance(item, dict):
            continue
        hard_min = float(spec["hard_min"])
        hard_max = float(spec["hard_max"])
        step = float(spec["step"])
        active_min = to_float(item.get("active_min"))
        active_max = to_float(item.get("active_max"))
        if active_min is None or active_max is None:
            active_min = float(spec["min"])
            active_max = float(spec["max"])
        active_min, active_max = sanitize_range(active_min, active_max, hard_min, hard_max, step)
        state["params"][name].update(
            {
                "hard_min": hard_min,
                "hard_max": hard_max,
                "active_min": active_min,
                "active_max": active_max,
                "step": step,
                "weight": max(0.05, min(20.0, float(to_float(item.get("weight")) or 1.0))),
                "keep_count": int(to_float(item.get("keep_count")) or 0),
                "discard_count": int(to_float(item.get("discard_count")) or 0),
                "crash_count": int(to_float(item.get("crash_count")) or 0),
            }
        )

    return state


def load_or_init_space_state(
    path: pathlib.Path,
    tunables: dict[str, dict[str, float]],
    profile_raw: dict[str, Any],
    campaign_id: str,
    profile_path: pathlib.Path,
    profile_hash: str,
    reuse_space_state: bool,
    reset_space_state: bool,
) -> tuple[dict[str, Any], str]:
    fresh = init_space_state(tunables, profile_raw, campaign_id, profile_path, profile_hash)

    if reset_space_state:
        return fresh, "reset_requested"
    if not path.exists():
        return fresh, "new_state"

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fresh, "invalid_json"

    loaded_campaign = loaded.get("campaign_id")
    loaded_profile_hash = loaded.get("profile_hash")
    loaded_profile_path = loaded.get("profile_path")

    mismatch = (
        loaded_campaign != campaign_id
        or loaded_profile_hash != profile_hash
        or str(loaded_profile_path) != str(profile_path)
    )

    if mismatch and not reuse_space_state:
        return fresh, "metadata_mismatch_reset"

    merged = _merge_loaded_params(fresh, loaded, tunables, profile_raw)
    return merged, "reused_state" if not mismatch else "metadata_mismatch_reused"


def save_space_state(path: pathlib.Path, state: dict[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_write_json(path, payload)


def build_adapted_profile(base_profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    out = {
        "strategy_file": base_profile["strategy_file"],
        "tunables": {},
    }
    for name, raw in base_profile["tunables"].items():
        param_state = state["params"][name]
        merged = dict(raw)
        merged["min"] = float(param_state["active_min"])
        merged["max"] = float(param_state["active_max"])
        merged["step"] = float(param_state["step"])
        merged["hard_min"] = float(param_state["hard_min"])
        merged["hard_max"] = float(param_state["hard_max"])
        merged["weight"] = float(param_state["weight"])
        out["tunables"][name] = merged
    return out


def ensure_current_inside_space(param_state: dict[str, Any], current: float) -> None:
    hard_min = float(param_state["hard_min"])
    hard_max = float(param_state["hard_max"])
    step = float(param_state["step"])
    lo = min(float(param_state["active_min"]), current)
    hi = max(float(param_state["active_max"]), current)
    lo, hi = sanitize_range(lo, hi, hard_min, hard_max, step)
    param_state["active_min"] = lo
    param_state["active_max"] = hi


def recenter_space(param_state: dict[str, Any], center: float, scale: float, min_span_steps: int) -> None:
    hard_min = float(param_state["hard_min"])
    hard_max = float(param_state["hard_max"])
    step = float(param_state["step"])
    hard_span = hard_max - hard_min
    if hard_span <= 0:
        return

    cur_span = max(float(param_state["active_max"]) - float(param_state["active_min"]), step)
    target_span = max(step * float(min_span_steps), cur_span * scale)
    target_span = min(target_span, hard_span)
    lo = center - target_span / 2.0
    hi = center + target_span / 2.0
    lo, hi = sanitize_range(lo, hi, hard_min, hard_max, step)
    param_state["active_min"] = lo
    param_state["active_max"] = hi


def read_last_matching_row(
    tsv_path: pathlib.Path,
    campaign_id: str,
    candidate_id: str,
    stage: str,
) -> dict[str, str] | None:
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return None
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        last = None
        for row in reader:
            if row.get("campaign_id") != campaign_id:
                continue
            if row.get("candidate_id") != candidate_id:
                continue
            if row.get("stage") != stage:
                continue
            last = row
        return last


def has_campaign_rows(tsv_path: pathlib.Path, campaign_id: str) -> bool:
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return False
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("campaign_id") == campaign_id:
                return True
    return False


def choose_final_suggestion(
    train_suggestion: str,
    holdout_suggestion: str | None,
    train_returncode: int,
    holdout_returncode: int | None,
) -> str:
    if train_returncode != 0:
        return "crash"
    if train_suggestion != "keep":
        return train_suggestion or "discard"
    if holdout_returncode is None:
        return "crash"
    if holdout_returncode != 0:
        return "crash"
    return holdout_suggestion or "discard"


def restore_strategy(strategy_file: pathlib.Path, text: str) -> None:
    strategy_file.write_text(text, encoding="utf-8")


def read_optional_text(path: pathlib.Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def restore_optional_file(path: pathlib.Path, text: str | None) -> None:
    if text is None:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_freqtrade_cmd(freqtrade_dir: pathlib.Path, freqtrade_bin: str | None) -> list[str]:
    if freqtrade_bin:
        tokens = shlex.split(freqtrade_bin)
        if len(tokens) == 1:
            candidate = pathlib.Path(tokens[0]).expanduser()
            if candidate.exists():
                return [str(candidate.resolve())]
            in_path = shutil.which(tokens[0])
            if in_path:
                return [in_path]
        return tokens

    local_candidates = [
        freqtrade_dir / ".venv" / "bin" / "freqtrade",
        freqtrade_dir / ".env" / "bin" / "freqtrade",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return [str(candidate)]

    in_path = shutil.which("freqtrade")
    if in_path:
        return [in_path]

    return [sys.executable, "-m", "freqtrade"]


def run_stream_cmd(cmd: list[str], cwd: pathlib.Path) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def should_run_hyperopt(iteration: int, cadence: int) -> bool:
    return iteration == 1 or ((iteration - 1) % cadence == 0)


def run_hyperopt(
    freqtrade_dir: pathlib.Path,
    freqtrade_bin: str | None,
    config: str,
    strategy: str,
    strategy_path: str | None,
    freqaimodel: str | None,
    freqaimodel_path: str | None,
    timerange: str,
    epochs: int,
    spaces: str,
    loss: str,
    jobs: int,
    min_trades: int,
    random_state: int,
    ignore_missing_spaces: bool,
) -> int:
    freqtrade_cmd = resolve_freqtrade_cmd(freqtrade_dir, freqtrade_bin)
    cmd = freqtrade_cmd + [
        "hyperopt",
        "--config",
        config,
        "--strategy",
        strategy,
        "--timerange",
        timerange,
        "-e",
        str(epochs),
        "--spaces",
        *shlex.split(spaces),
        "--hyperopt-loss",
        loss,
        "-j",
        str(jobs),
        "--min-trades",
        str(min_trades),
        "--random-state",
        str(random_state),
    ]
    if ignore_missing_spaces:
        cmd.append("--ignore-missing-spaces")
    if strategy_path:
        cmd.extend(["--strategy-path", strategy_path])
    if freqaimodel:
        cmd.extend(["--freqaimodel", freqaimodel])
    if freqaimodel_path:
        cmd.extend(["--freqaimodel-path", freqaimodel_path])
    return run_stream_cmd(cmd, cwd=freqtrade_dir)


def run_llm_patch_stage(
    *,
    repo_dir: pathlib.Path,
    model_file: pathlib.Path,
    campaign_id: str,
    results_tsv: pathlib.Path,
    review_log: pathlib.Path,
    llm_model: str,
    credentials_file: pathlib.Path,
    context_rows: int,
    timeout_sec: int,
    repair_attempts: int,
    upstream_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    allowed_rel_path = "freqtrade/freqaimodels/AutoresearchLSTMRegressor.py"
    base_source = read_strategy_text(model_file)
    max_attempts = max(1, repair_attempts + 1)

    info: dict[str, Any] = {
        "ran": True,
        "model": llm_model,
        "request_id": None,
        "prompt_hash": None,
        "patch_applied": False,
        "repair_attempts_used": 0,
        "error": None,
        "patch_text": "",
        "usage": None,
        "upstream_ref": upstream_bundle.get("ref") if isinstance(upstream_bundle, dict) else None,
        "upstream_commit": upstream_bundle.get("commit") if isinstance(upstream_bundle, dict) else None,
    }

    repair_hint: str | None = None
    previous_patch: str | None = None

    for attempt in range(1, max_attempts + 1):
        info["repair_attempts_used"] = attempt - 1
        restore_strategy(model_file, base_source)

        try:
            generated = llm_patch_engine.generate_patch_with_openai(
                credentials_file=credentials_file,
                model=llm_model,
                target_relative_path=allowed_rel_path,
                target_source=base_source,
                results_tsv=results_tsv,
                review_log=review_log,
                campaign_id=campaign_id,
                context_rows=context_rows,
                timeout_sec=timeout_sec,
                upstream_bundle=upstream_bundle,
                repair_hint=repair_hint,
                previous_patch=previous_patch,
            )
        except Exception as exc:
            info["error"] = f"llm_request_failed: {exc}"
            repair_hint = str(info["error"])
            continue

        patch_text = str(generated.get("patch_text") or "")
        previous_patch = patch_text
        info["patch_text"] = patch_text
        info["request_id"] = generated.get("request_id")
        info["prompt_hash"] = generated.get("prompt_hash")
        info["usage"] = generated.get("usage")

        if not patch_text.strip():
            info["error"] = "empty_patch_from_llm"
            repair_hint = str(info["error"])
            continue

        ok, target_error, _ = llm_patch_engine.validate_patch_targets(patch_text, allowed_rel_path)
        if not ok:
            info["error"] = target_error
            repair_hint = str(target_error)
            continue

        ok, apply_error = llm_patch_engine.apply_unified_diff(repo_dir, patch_text)
        if not ok:
            info["error"] = apply_error
            repair_hint = str(apply_error)
            continue

        ok, compile_error = llm_patch_engine.run_py_compile(model_file)
        if not ok:
            info["error"] = compile_error
            repair_hint = str(compile_error)
            continue

        info["patch_applied"] = True
        info["error"] = None
        info["repair_attempts_used"] = attempt - 1
        return info

    restore_strategy(model_file, base_source)
    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoresearch loop for FreqAI strategy tuning")
    parser.add_argument("--freqtrade-dir", required=True, help="Path to Freqtrade project")
    parser.add_argument("--config", required=True, help="Freqtrade config path (as seen from freqtrade-dir)")
    parser.add_argument("--freqtrade-bin", default=None, help="Freqtrade executable (or command), e.g. /path/.venv/bin/freqtrade")
    parser.add_argument("--strategy", default="AutoresearchFreqAIStrategy", help="Strategy class name")
    parser.add_argument("--strategy-path", default=None, help="Strategy path for freqtrade")
    parser.add_argument("--freqaimodel", default="AutoresearchLSTMRegressor", help="FreqAI model class")
    parser.add_argument("--freqaimodel-path", default=None, help="Optional lookup path for custom FreqAI models")
    parser.add_argument("--train-timerange", required=True, help="Training/optimization timerange")
    parser.add_argument("--holdout-timerange", required=True, help="Holdout validation timerange")
    parser.add_argument("--iterations", type=int, default=20, help="Number of experiments to run")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--profile", default="freqtrade/autoresearch_profile.example.json", help="Mutation profile JSON")
    parser.add_argument("--save-profile", default=None, help="Write adapted profile for next runs")
    parser.add_argument("--space-state", default=None, help="Adaptive space state JSON path")
    parser.add_argument("--campaign-id", default="default", help="Campaign identifier")
    parser.add_argument("--reuse-space-state", action="store_true", help="Allow metadata-mismatched state reuse")
    parser.add_argument("--reset-space-state", action="store_true", help="Reset state before run")
    parser.add_argument("--lock-file", default=None, help="Lock file path (default: campaign-scoped in freqtrade/runs)")

    parser.add_argument("--no-space-adaptation", action="store_true", help="Disable automatic search-space adaptation")
    parser.add_argument("--keep-shrink", type=float, default=0.85, help="Range scale after keep (<1 shrinks)")
    parser.add_argument("--discard-expand", type=float, default=1.10, help="Range scale after discard (>1 expands)")
    parser.add_argument("--crash-expand", type=float, default=1.25, help="Range scale after crash (>1 expands)")
    parser.add_argument("--min-span-steps", type=int, default=6, help="Minimum active range width in step units")
    parser.add_argument("--repo-dir", default=None, help="Repo root of this project")
    parser.add_argument("--results-tsv", default=None, help="TSV results file path")
    parser.add_argument("--dd-penalty", type=float, default=0.5, help="Score penalty on drawdown")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Required score improvement to keep")
    parser.add_argument("--baseline-if-empty", action="store_true", help="Run baseline train+holdout if campaign has no rows")
    parser.add_argument("--weight-up", type=float, default=1.2, help="Multiplier for parameter weight after keep")
    parser.add_argument("--weight-down", type=float, default=0.9, help="Multiplier for parameter weight after discard")
    parser.add_argument("--weight-crash", type=float, default=0.75, help="Multiplier for parameter weight after crash")
    parser.add_argument("--review-log", default=None, help="Optional JSONL file for per-iteration review records")
    parser.add_argument("--hyperopt-cadence", type=int, default=5, help="Run hyperopt every N iterations (1, 1+N, ...)")
    parser.add_argument("--hyperopt-epochs", type=int, default=100, help="Hyperopt epochs per run")
    parser.add_argument("--hyperopt-spaces", default="all", help="Hyperopt spaces argument, e.g. 'all' or 'buy sell'")
    parser.add_argument(
        "--hyperopt-loss",
        default="ProfitDrawDownHyperOptLoss",
        help="Hyperopt loss class name",
    )
    parser.add_argument("--hyperopt-jobs", type=int, default=-1, help="Hyperopt worker jobs")
    parser.add_argument("--hyperopt-min-trades", type=int, default=1, help="Hyperopt min trades filter")
    parser.add_argument(
        "--hyperopt-ignore-missing-spaces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --ignore-missing-spaces to freqtrade hyperopt",
    )
    parser.add_argument("--llm-enable", action="store_true", help="Enable LLM-generated model patching each iteration")
    parser.add_argument("--llm-model", default="gpt-5-mini", help="OpenAI model used for patch generation")
    parser.add_argument(
        "--llm-credentials-file",
        default=str(llm_patch_engine.DEFAULT_CREDENTIALS_PATH),
        help="Credentials JSON path (default: ~/.config/autoresearch/openai.json)",
    )
    parser.add_argument("--llm-repair-attempts", type=int, default=2, help="Automatic LLM repair retries after failed patch checks")
    parser.add_argument("--llm-context-rows", type=int, default=10, help="Recent results/review rows to include in LLM context")
    parser.add_argument("--llm-timeout-sec", type=int, default=60, help="HTTP timeout for LLM requests")
    parser.add_argument("--llm-sync-upstream", action="store_true", help="Fetch optional karpathy/autoresearch context snippets")
    parser.add_argument("--llm-upstream-ref", default="main", help="Upstream ref used when --llm-sync-upstream is set")

    parser.add_argument("--min-profit-pct", type=float, default=0.0, help="Absolute gate: minimum total profit %%")
    parser.add_argument("--max-drawdown-pct", type=float, default=12.0, help="Absolute gate: maximum drawdown %%")
    parser.add_argument("--min-sharpe", type=float, default=0.0, help="Absolute gate: minimum Sharpe")
    parser.add_argument("--pair-min-trades-floor", type=int, default=10, help="Floor for pair-coverage min trades")
    parser.add_argument(
        "--pair-min-trades-mode",
        default="dynamic",
        choices=["dynamic", "off"],
        help="Pair-coverage mode (dynamic/off)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve() if args.repo_dir else pathlib.Path(__file__).resolve().parents[1]
    freqtrade_dir = pathlib.Path(args.freqtrade_dir).expanduser().resolve()
    profile_path = pathlib.Path(args.profile).expanduser()
    if not profile_path.is_absolute():
        profile_path = (repo_dir / profile_path).resolve()
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    safe_strategy = re.sub(r"[^A-Za-z0-9._-]+", "_", args.strategy)
    save_profile_path = pathlib.Path(args.save_profile).expanduser().resolve() if args.save_profile else (
        repo_dir / "freqtrade" / f"autoresearch_profile.{args.campaign_id}.json"
    )

    space_state_path = pathlib.Path(args.space_state).expanduser().resolve() if args.space_state else (
        repo_dir / "freqtrade" / "runs" / f"space_state_{safe_strategy}_{args.campaign_id}.json"
    )

    results_tsv = pathlib.Path(args.results_tsv).expanduser().resolve() if args.results_tsv else (
        repo_dir / "freqtrade" / "results" / f"results_{args.campaign_id}.tsv"
    )

    review_log = pathlib.Path(args.review_log).expanduser().resolve() if args.review_log else (
        repo_dir
        / "freqtrade"
        / "runs"
        / f"review_{args.campaign_id}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    review_log.parent.mkdir(parents=True, exist_ok=True)

    lock_path = pathlib.Path(args.lock_file).expanduser().resolve() if args.lock_file else (
        repo_dir / "freqtrade" / "runs" / f"lock_{safe_strategy}_{args.campaign_id}.lock"
    )

    if args.keep_shrink <= 0 or args.discard_expand <= 0 or args.crash_expand <= 0:
        raise ValueError("Space adaptation scales must be > 0.")
    if args.min_span_steps < 1:
        raise ValueError("--min-span-steps must be >= 1.")
    if args.hyperopt_cadence < 1:
        raise ValueError("--hyperopt-cadence must be >= 1.")
    if args.hyperopt_epochs < 1:
        raise ValueError("--hyperopt-epochs must be >= 1.")
    if args.llm_repair_attempts < 0:
        raise ValueError("--llm-repair-attempts must be >= 0.")
    if args.llm_context_rows < 1:
        raise ValueError("--llm-context-rows must be >= 1.")
    if args.llm_timeout_sec < 1:
        raise ValueError("--llm-timeout-sec must be >= 1.")

    rng = random.Random(args.seed)
    profile = load_profile(profile_path)
    profile_hash = profile_fingerprint(profile)
    tunables: dict[str, dict[str, float]] = profile["_normalized_tunables"]

    strategy_file = pathlib.Path(profile["strategy_file"])
    if not strategy_file.is_absolute():
        strategy_file = (repo_dir / strategy_file).resolve()
    strategy_param_file = strategy_file.with_suffix(".json")
    llm_model_file = (repo_dir / "freqtrade" / "freqaimodels" / "AutoresearchLSTMRegressor.py").resolve()
    llm_credentials_file = pathlib.Path(args.llm_credentials_file).expanduser().resolve()
    if args.llm_enable and not llm_model_file.exists():
        raise FileNotFoundError(f"LLM target model file not found: {llm_model_file}")

    upstream_bundle: dict[str, Any] | None = None
    if args.llm_enable and args.llm_sync_upstream:
        upstream_cache_dir = repo_dir / "freqtrade" / "runs" / f"upstream_cache_{args.campaign_id}"
        upstream_bundle = llm_patch_engine.fetch_upstream_context_bundle(
            cache_dir=upstream_cache_dir,
            ref=args.llm_upstream_ref,
            timeout_sec=args.llm_timeout_sec,
        )
        print(
            "Upstream context:"
            f" ref={upstream_bundle.get('ref')} commit={upstream_bundle.get('commit')} error={upstream_bundle.get('error')}",
            flush=True,
        )

    interrupted = {"signal": None}

    def _signal_handler(signum: int, _frame: Any) -> None:
        interrupted["signal"] = signum
        raise KeyboardInterrupt()

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)

    try:
        with exclusive_lock(lock_path):
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)

            space_state, state_reason = load_or_init_space_state(
                path=space_state_path,
                tunables=tunables,
                profile_raw=profile,
                campaign_id=args.campaign_id,
                profile_path=profile_path,
                profile_hash=profile_hash,
                reuse_space_state=args.reuse_space_state,
                reset_space_state=args.reset_space_state,
            )
            print(f"Space state: {state_reason} ({space_state_path})", flush=True)

            param_weights = {name: float(space_state["params"][name]["weight"]) for name in tunables}
            save_space_state(space_state_path, space_state)
            if not args.no_space_adaptation:
                adapted_profile = build_adapted_profile(profile, space_state)
                _atomic_write_json(save_profile_path, adapted_profile)

            committed_text = read_strategy_text(strategy_file)
            committed_param_text = read_optional_text(strategy_param_file)
            committed_model_text = read_strategy_text(llm_model_file) if args.llm_enable else None

            if args.baseline_if_empty and not has_campaign_rows(results_tsv, args.campaign_id):
                baseline_candidate = "baseline"
                print("Running baseline hyperopt stage...", flush=True)
                baseline_hyperopt_rc = run_hyperopt(
                    freqtrade_dir=freqtrade_dir,
                    freqtrade_bin=args.freqtrade_bin,
                    config=args.config,
                    strategy=args.strategy,
                    strategy_path=args.strategy_path,
                    freqaimodel=args.freqaimodel,
                    freqaimodel_path=args.freqaimodel_path,
                    timerange=args.train_timerange,
                    epochs=args.hyperopt_epochs,
                    spaces=args.hyperopt_spaces,
                    loss=args.hyperopt_loss,
                    jobs=args.hyperopt_jobs,
                    min_trades=args.hyperopt_min_trades,
                    random_state=args.seed,
                    ignore_missing_spaces=args.hyperopt_ignore_missing_spaces,
                )
                if baseline_hyperopt_rc != 0:
                    print("Baseline hyperopt failed. Stopping.", flush=True)
                    restore_strategy(strategy_file, committed_text)
                    restore_optional_file(strategy_param_file, committed_param_text)
                    return baseline_hyperopt_rc
                committed_param_text = read_optional_text(strategy_param_file)
                print("Running baseline train stage...", flush=True)
                train_proc = run_backtest_runner(
                    repo_dir=repo_dir,
                    freqtrade_dir=freqtrade_dir,
                    config=args.config,
                    freqtrade_bin=args.freqtrade_bin,
                    strategy=args.strategy,
                    strategy_path=args.strategy_path,
                    freqaimodel=args.freqaimodel,
                    freqaimodel_path=args.freqaimodel_path,
                    timerange=args.train_timerange,
                    description="baseline-train",
                    results_tsv=results_tsv,
                    dd_penalty=args.dd_penalty,
                    min_improvement=args.min_improvement,
                    campaign_id=args.campaign_id,
                    candidate_id=baseline_candidate,
                    stage="train",
                    min_profit_pct=args.min_profit_pct,
                    max_drawdown_pct=args.max_drawdown_pct,
                    min_sharpe=args.min_sharpe,
                    pair_min_trades_floor=args.pair_min_trades_floor,
                    pair_min_trades_mode=args.pair_min_trades_mode,
                )
                if train_proc.returncode != 0:
                    print("Baseline train failed. Stopping.", flush=True)
                    return train_proc.returncode
                train_row = read_last_matching_row(results_tsv, args.campaign_id, baseline_candidate, "train") or {}
                if train_row.get("suggestion") == "keep":
                    print("Running baseline holdout stage...", flush=True)
                    holdout_proc = run_backtest_runner(
                        repo_dir=repo_dir,
                        freqtrade_dir=freqtrade_dir,
                        config=args.config,
                        freqtrade_bin=args.freqtrade_bin,
                        strategy=args.strategy,
                        strategy_path=args.strategy_path,
                        freqaimodel=args.freqaimodel,
                        freqaimodel_path=args.freqaimodel_path,
                        timerange=args.holdout_timerange,
                        description="baseline-holdout",
                        results_tsv=results_tsv,
                        dd_penalty=args.dd_penalty,
                        min_improvement=args.min_improvement,
                        campaign_id=args.campaign_id,
                        candidate_id=baseline_candidate,
                        stage="holdout",
                        min_profit_pct=args.min_profit_pct,
                        max_drawdown_pct=args.max_drawdown_pct,
                        min_sharpe=args.min_sharpe,
                        pair_min_trades_floor=args.pair_min_trades_floor,
                        pair_min_trades_mode=args.pair_min_trades_mode,
                    )
                    if holdout_proc.returncode != 0:
                        print("Baseline holdout failed. Stopping.", flush=True)
                        return holdout_proc.returncode

            for i in range(1, args.iterations + 1):
                if interrupted["signal"] is not None:
                    break

                candidate_id = f"cand_{i:04d}_{dt.datetime.now(dt.timezone.utc).strftime('%H%M%S')}"
                strategy_mutation_enabled = not args.llm_enable
                param: str | None = None
                current: float | None = None
                proposed: float | None = None
                before_space: list[float] | None = None
                after_space: list[float] | None = None
                param_state: dict[str, Any] | None = None
                desc = "llm_patch: AutoresearchLSTMRegressor.py"

                llm_info: dict[str, Any] = {
                    "ran": False,
                    "model": args.llm_model,
                    "request_id": None,
                    "prompt_hash": None,
                    "patch_applied": False,
                    "repair_attempts_used": 0,
                    "error": None,
                    "patch_text": "",
                    "usage": None,
                    "upstream_ref": None,
                    "upstream_commit": None,
                }

                if strategy_mutation_enabled:
                    current_text = read_strategy_text(strategy_file)
                    param = weighted_choice(param_weights, rng)
                    param_state = space_state["params"][param]
                    current = parse_value(current_text, param)
                    ensure_current_inside_space(param_state, current)

                    spec = {
                        "min": float(param_state["active_min"]),
                        "max": float(param_state["active_max"]),
                        "step": float(param_state["step"]),
                    }
                    proposed = mutate_value(current, spec, rng)
                    if proposed == current:
                        print(f"[{i:03d}] skip {param}: no mutation possible", flush=True)
                        continue

                    before_space = [float(param_state["active_min"]), float(param_state["active_max"])]
                    desc = f"{param}: {current} -> {proposed}"

                    mutated_text = replace_assignment(current_text, param, proposed)
                    mutated_text = enforce_consistency(mutated_text)
                    strategy_file.write_text(mutated_text, encoding="utf-8")
                else:
                    mutated_text = read_strategy_text(strategy_file)

                llm_stage_ok = True
                if args.llm_enable:
                    print(f"[{i:03d}] llm_patch {candidate_id} {desc}", flush=True)
                    llm_info = run_llm_patch_stage(
                        repo_dir=repo_dir,
                        model_file=llm_model_file,
                        campaign_id=args.campaign_id,
                        results_tsv=results_tsv,
                        review_log=review_log,
                        llm_model=args.llm_model,
                        credentials_file=llm_credentials_file,
                        context_rows=args.llm_context_rows,
                        timeout_sec=args.llm_timeout_sec,
                        repair_attempts=args.llm_repair_attempts,
                        upstream_bundle=upstream_bundle,
                    )
                    llm_stage_ok = bool(llm_info.get("patch_applied"))

                hyperopt_ran = should_run_hyperopt(i, args.hyperopt_cadence) if llm_stage_ok else False
                hyperopt_returncode: int | None = None
                if llm_stage_ok and hyperopt_ran:
                    print(f"[{i:03d}] hyperopt {candidate_id} {desc}", flush=True)
                    hyperopt_returncode = run_hyperopt(
                        freqtrade_dir=freqtrade_dir,
                        freqtrade_bin=args.freqtrade_bin,
                        config=args.config,
                        strategy=args.strategy,
                        strategy_path=args.strategy_path,
                        freqaimodel=args.freqaimodel,
                        freqaimodel_path=args.freqaimodel_path,
                        timerange=args.train_timerange,
                        epochs=args.hyperopt_epochs,
                        spaces=args.hyperopt_spaces,
                        loss=args.hyperopt_loss,
                        jobs=args.hyperopt_jobs,
                        min_trades=args.hyperopt_min_trades,
                        random_state=args.seed + i,
                        ignore_missing_spaces=args.hyperopt_ignore_missing_spaces,
                    )

                train_row: dict[str, str] = {}
                holdout_proc: subprocess.CompletedProcess[str] | None = None
                holdout_row: dict[str, str] = {}
                holdout_suggestion: str | None = None

                if not llm_stage_ok:
                    print(f"[{i:03d}] llm patch crash ({llm_info.get('error')})", flush=True)
                    train_proc = subprocess.CompletedProcess(args=["llm_patch"], returncode=1)
                    train_suggestion = "crash"
                elif hyperopt_returncode is not None and hyperopt_returncode != 0:
                    print(f"[{i:03d}] hyperopt crash (rc={hyperopt_returncode})", flush=True)
                    train_proc = subprocess.CompletedProcess(args=["hyperopt"], returncode=hyperopt_returncode)
                    train_suggestion = "crash"
                else:
                    if before_space is not None:
                        print(
                            f"[{i:03d}] train {candidate_id} {desc} within {before_space[0]}..{before_space[1]}",
                            flush=True,
                        )
                    else:
                        print(f"[{i:03d}] train {candidate_id} {desc}", flush=True)
                    train_proc = run_backtest_runner(
                        repo_dir=repo_dir,
                        freqtrade_dir=freqtrade_dir,
                        config=args.config,
                        freqtrade_bin=args.freqtrade_bin,
                        strategy=args.strategy,
                        strategy_path=args.strategy_path,
                        freqaimodel=args.freqaimodel,
                        freqaimodel_path=args.freqaimodel_path,
                        timerange=args.train_timerange,
                        description=f"{desc} [train]",
                        results_tsv=results_tsv,
                        dd_penalty=args.dd_penalty,
                        min_improvement=args.min_improvement,
                        campaign_id=args.campaign_id,
                        candidate_id=candidate_id,
                        stage="train",
                        min_profit_pct=args.min_profit_pct,
                        max_drawdown_pct=args.max_drawdown_pct,
                        min_sharpe=args.min_sharpe,
                        pair_min_trades_floor=args.pair_min_trades_floor,
                        pair_min_trades_mode=args.pair_min_trades_mode,
                    )

                    train_row = read_last_matching_row(results_tsv, args.campaign_id, candidate_id, "train") or {}
                    train_suggestion = train_row.get("suggestion", "crash") if train_proc.returncode == 0 else "crash"

                    if train_proc.returncode == 0 and train_suggestion == "keep":
                        print(f"[{i:03d}] holdout {candidate_id} {desc}", flush=True)
                        holdout_proc = run_backtest_runner(
                            repo_dir=repo_dir,
                            freqtrade_dir=freqtrade_dir,
                            config=args.config,
                            freqtrade_bin=args.freqtrade_bin,
                            strategy=args.strategy,
                            strategy_path=args.strategy_path,
                            freqaimodel=args.freqaimodel,
                            freqaimodel_path=args.freqaimodel_path,
                            timerange=args.holdout_timerange,
                            description=f"{desc} [holdout]",
                            results_tsv=results_tsv,
                            dd_penalty=args.dd_penalty,
                            min_improvement=args.min_improvement,
                            campaign_id=args.campaign_id,
                            candidate_id=candidate_id,
                            stage="holdout",
                            min_profit_pct=args.min_profit_pct,
                            max_drawdown_pct=args.max_drawdown_pct,
                            min_sharpe=args.min_sharpe,
                            pair_min_trades_floor=args.pair_min_trades_floor,
                            pair_min_trades_mode=args.pair_min_trades_mode,
                        )
                        holdout_row = read_last_matching_row(results_tsv, args.campaign_id, candidate_id, "holdout") or {}
                        holdout_suggestion = holdout_row.get("suggestion", "crash") if holdout_proc.returncode == 0 else "crash"

                final_suggestion = choose_final_suggestion(
                    train_suggestion=train_suggestion,
                    holdout_suggestion=holdout_suggestion,
                    train_returncode=train_proc.returncode,
                    holdout_returncode=(holdout_proc.returncode if holdout_proc is not None else None),
                )

                if final_suggestion == "keep":
                    committed_text = read_strategy_text(strategy_file)
                    committed_param_text = read_optional_text(strategy_param_file)
                    if args.llm_enable and committed_model_text is not None:
                        committed_model_text = read_strategy_text(llm_model_file)
                    if strategy_mutation_enabled and param_state is not None and param is not None and proposed is not None:
                        param_weights[param] *= args.weight_up
                        param_state["keep_count"] = int(param_state["keep_count"]) + 1
                        if not args.no_space_adaptation:
                            recenter_space(param_state, proposed, args.keep_shrink, args.min_span_steps)
                    print(f"[{i:03d}] KEEP (candidate={candidate_id})", flush=True)
                elif final_suggestion == "discard":
                    restore_strategy(strategy_file, committed_text)
                    restore_optional_file(strategy_param_file, committed_param_text)
                    if args.llm_enable and committed_model_text is not None:
                        restore_strategy(llm_model_file, committed_model_text)
                    if strategy_mutation_enabled and param_state is not None and param is not None and current is not None:
                        param_weights[param] *= args.weight_down
                        param_state["discard_count"] = int(param_state["discard_count"]) + 1
                        if not args.no_space_adaptation:
                            recenter_space(param_state, current, args.discard_expand, args.min_span_steps)
                    print(f"[{i:03d}] DISCARD -> reverted change", flush=True)
                else:
                    restore_strategy(strategy_file, committed_text)
                    restore_optional_file(strategy_param_file, committed_param_text)
                    if args.llm_enable and committed_model_text is not None:
                        restore_strategy(llm_model_file, committed_model_text)
                    if strategy_mutation_enabled and param_state is not None and param is not None and current is not None:
                        param_weights[param] *= args.weight_crash
                        param_state["crash_count"] = int(param_state["crash_count"]) + 1
                        if not args.no_space_adaptation:
                            recenter_space(param_state, current, args.crash_expand, args.min_span_steps)
                    print(f"[{i:03d}] CRASH -> reverted change", flush=True)

                if strategy_mutation_enabled:
                    for name in param_weights:
                        param_weights[name] = min(max(param_weights[name], 0.05), 20.0)
                        space_state["params"][name]["weight"] = float(param_weights[name])
                    if param_state is not None:
                        after_space = [float(param_state["active_min"]), float(param_state["active_max"])]
                    if not args.no_space_adaptation:
                        adapted_profile = build_adapted_profile(profile, space_state)
                        _atomic_write_json(save_profile_path, adapted_profile)

                save_space_state(space_state_path, space_state)

                review_record = {
                    "iteration": i,
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "campaign_id": args.campaign_id,
                    "candidate_id": candidate_id,
                    "param": param,
                    "from": current,
                    "to": proposed,
                    "space_before": before_space,
                    "space_after": after_space,
                    "llm": llm_info,
                    "hyperopt": {
                        "ran": hyperopt_ran,
                        "returncode": hyperopt_returncode,
                        "cadence": args.hyperopt_cadence,
                        "epochs": args.hyperopt_epochs,
                        "spaces": args.hyperopt_spaces,
                        "loss": args.hyperopt_loss,
                    },
                    "train": {
                        "status": train_row.get("status"),
                        "suggestion": train_row.get("suggestion"),
                        "decision_reason": train_row.get("decision_reason"),
                        "score": to_float(train_row.get("score")),
                    },
                    "holdout": {
                        "status": holdout_row.get("status"),
                        "suggestion": holdout_row.get("suggestion"),
                        "decision_reason": holdout_row.get("decision_reason"),
                        "score": to_float(holdout_row.get("score")),
                    }
                    if holdout_row
                    else None,
                    "final_suggestion": final_suggestion,
                    "param_weight": float(param_weights[param]) if (param and param in param_weights) else None,
                    "description": desc,
                }
                with review_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(review_record) + "\n")

            print(f"Autoresearch loop finished. Review log: {review_log}", flush=True)
            print(f"Space state saved: {space_state_path}", flush=True)
            if not args.no_space_adaptation:
                print(f"Adapted profile saved: {save_profile_path}", flush=True)

            restore_strategy(strategy_file, committed_text)
            restore_optional_file(strategy_param_file, committed_param_text)
            if args.llm_enable and committed_model_text is not None:
                restore_strategy(llm_model_file, committed_model_text)
            save_space_state(space_state_path, space_state)

    except RuntimeError as exc:
        print(f"Lock error: {exc}", flush=True)
        return 2
    except KeyboardInterrupt:
        try:
            # Best-effort restore after interrupt.
            if "strategy_file" in locals() and "committed_text" in locals():
                restore_strategy(strategy_file, committed_text)
            if "strategy_param_file" in locals() and "committed_param_text" in locals():
                restore_optional_file(strategy_param_file, committed_param_text)
            if "llm_model_file" in locals() and "committed_model_text" in locals() and committed_model_text is not None:
                restore_strategy(llm_model_file, committed_model_text)
            if "space_state_path" in locals() and "space_state" in locals():
                save_space_state(space_state_path, space_state)
        finally:
            signame = interrupted["signal"]
            if signame is None:
                print("Interrupted. Restored strategy/model files and saved state.", flush=True)
            else:
                print(f"Interrupted by signal {signame}. Restored strategy/model files and saved state.", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
