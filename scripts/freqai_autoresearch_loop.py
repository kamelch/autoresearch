#!/usr/bin/env python3
"""Autoresearch-style loop for FreqAI strategy optimization.

This script:
- mutates strategy class parameters (one change per iteration)
- runs one backtest via scripts/run_freqtrade_backtest.py
- reads keep/discard/crash suggestion from results.tsv
- keeps winning edits and auto-reverts losers
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import random
import re
import subprocess
import sys
from typing import Any


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


def load_profile(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "strategy_file" not in data or "tunables" not in data:
        raise ValueError("Profile must contain 'strategy_file' and 'tunables'.")
    if not isinstance(data["tunables"], dict) or not data["tunables"]:
        raise ValueError("'tunables' must be a non-empty object.")
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


def decimals_for_step(step: float) -> int:
    txt = f"{step:.12f}".rstrip("0").rstrip(".")
    if "." not in txt:
        return 0
    return len(txt.split(".", 1)[1])


def quantize(value: float, step: float, decimals: int) -> float:
    q = round(round(value / step) * step, decimals)
    return q


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def mutate_value(current: float, spec: dict[str, Any], rng: random.Random) -> float:
    lo = float(spec["min"])
    hi = float(spec["max"])
    step = float(spec["step"])
    decimals = decimals_for_step(step)

    # Random step jump around current value.
    jump = rng.choice([-3, -2, -1, 1, 2, 3]) * step
    candidate = clip(current + jump, lo, hi)
    candidate = quantize(candidate, step, decimals)

    if candidate == current:
        # Fallback: random value on grid within bounds.
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
    return text[:m.start()] + f"{m.group('prefix')}{val_token}{m.group('suffix')}" + text[m.end():]


def read_last_row(tsv_path: pathlib.Path) -> dict[str, str] | None:
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return None
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        last = None
        for row in reader:
            last = row
        return last


def enforce_consistency(text: str) -> str:
    # Keep probability gate ordering sane.
    lp = parse_value(text, "long_prob_min")
    sp = parse_value(text, "short_prob_max")
    le = parse_value(text, "long_exit_prob_max")
    se = parse_value(text, "short_exit_prob_min")

    changed = False

    # short entry probability should be <= long entry probability.
    if sp >= lp:
        sp = max(0.0, lp - 0.01)
        text = replace_assignment(text, "short_prob_max", sp)
        changed = True

    # Long exit probability should not exceed long entry threshold.
    if le >= lp:
        le = max(0.0, lp - 0.01)
        text = replace_assignment(text, "long_exit_prob_max", le)
        changed = True

    # Short exit probability should not be below short entry threshold.
    if se <= sp:
        se = min(1.0, sp + 0.01)
        text = replace_assignment(text, "short_exit_prob_min", se)
        changed = True

    return text if changed else text


def run_backtest_runner(
    repo_dir: pathlib.Path,
    freqtrade_dir: pathlib.Path,
    config: str,
    strategy: str,
    strategy_path: str | None,
    freqaimodel: str | None,
    timerange: str,
    description: str,
    results_tsv: pathlib.Path,
    dd_penalty: float,
    min_improvement: float,
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
    ]
    if strategy_path:
        cmd.extend(["--strategy-path", strategy_path])
    if freqaimodel:
        cmd.extend(["--freqaimodel", freqaimodel])

    return subprocess.run(cmd, text=True, cwd=repo_dir, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Autoresearch loop for FreqAI strategy tuning")
    parser.add_argument("--freqtrade-dir", required=True, help="Path to Freqtrade project")
    parser.add_argument("--config", required=True, help="Freqtrade config path (as seen from freqtrade-dir)")
    parser.add_argument("--strategy", default="AutoresearchFreqAIStrategy", help="Strategy class name")
    parser.add_argument("--strategy-path", default=None, help="Strategy path for freqtrade")
    parser.add_argument("--freqaimodel", default="PyTorchMLPRegressor", help="FreqAI model class")
    parser.add_argument("--timerange", required=True, help="Backtest timerange")
    parser.add_argument("--iterations", type=int, default=20, help="Number of experiments to run")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--profile", default="freqtrade/autoresearch_profile.example.json", help="Mutation profile JSON")
    parser.add_argument("--repo-dir", default=None, help="Repo root of this project")
    parser.add_argument("--results-tsv", default=None, help="TSV results file path")
    parser.add_argument("--dd-penalty", type=float, default=0.5, help="Score penalty on drawdown")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Required score improvement to keep")
    parser.add_argument("--baseline-if-empty", action="store_true", help="Run one baseline first if results TSV has no rows")
    parser.add_argument("--weight-up", type=float, default=1.2, help="Multiplier for parameter weight after keep")
    parser.add_argument("--weight-down", type=float, default=0.9, help="Multiplier for parameter weight after discard")
    parser.add_argument("--weight-crash", type=float, default=0.75, help="Multiplier for parameter weight after crash")
    parser.add_argument("--review-log", default=None, help="Optional JSONL file for per-iteration review records")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve() if args.repo_dir else pathlib.Path(__file__).resolve().parents[1]
    freqtrade_dir = pathlib.Path(args.freqtrade_dir).expanduser().resolve()
    profile_path = pathlib.Path(args.profile).expanduser()
    if not profile_path.is_absolute():
        profile_path = (repo_dir / profile_path).resolve()

    results_tsv = pathlib.Path(args.results_tsv).expanduser().resolve() if args.results_tsv else (repo_dir / "freqtrade" / "results.tsv")
    review_log = pathlib.Path(args.review_log).expanduser().resolve() if args.review_log else (
        repo_dir / "freqtrade" / "runs" / f"review_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    review_log.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    profile = load_profile(profile_path)

    strategy_file = pathlib.Path(profile["strategy_file"])
    if not strategy_file.is_absolute():
        strategy_file = (repo_dir / strategy_file).resolve()
    tunables = profile["tunables"]
    param_weights = {name: 1.0 for name in tunables}

    base_text = read_strategy_text(strategy_file)

    def print_proc(proc: subprocess.CompletedProcess[str]) -> None:
        if proc.stdout.strip():
            print(proc.stdout.rstrip())
        if proc.stderr.strip():
            print("[stderr]")
            print(proc.stderr.rstrip())

    # Optional baseline for empty results file.
    last_row = read_last_row(results_tsv)
    if args.baseline_if_empty and last_row is None:
        print("Running baseline...")
        proc = run_backtest_runner(
            repo_dir=repo_dir,
            freqtrade_dir=freqtrade_dir,
            config=args.config,
            strategy=args.strategy,
            strategy_path=args.strategy_path,
            freqaimodel=args.freqaimodel,
            timerange=args.timerange,
            description="baseline",
            results_tsv=results_tsv,
            dd_penalty=args.dd_penalty,
            min_improvement=args.min_improvement,
        )
        print_proc(proc)
        if proc.returncode != 0:
            print("Baseline failed. Stopping.")
            return proc.returncode
        last_row = read_last_row(results_tsv)

    for i in range(1, args.iterations + 1):
        current_text = read_strategy_text(strategy_file)

        # AI-style adaptive choice: bias toward parameters with better historical impact.
        param = weighted_choice(param_weights, rng)
        spec = tunables[param]
        current = parse_value(current_text, param)
        proposed = mutate_value(current, spec, rng)
        if proposed == current:
            print(f"[{i:03d}] skip {param}: no mutation possible")
            continue

        mutated_text = replace_assignment(current_text, param, proposed)
        mutated_text = enforce_consistency(mutated_text)
        strategy_file.write_text(mutated_text, encoding="utf-8")

        desc = f"{param}: {current} -> {proposed}"
        print(f"[{i:03d}] testing {desc}")

        proc = run_backtest_runner(
            repo_dir=repo_dir,
            freqtrade_dir=freqtrade_dir,
            config=args.config,
            strategy=args.strategy,
            strategy_path=args.strategy_path,
            freqaimodel=args.freqaimodel,
            timerange=args.timerange,
            description=desc,
            results_tsv=results_tsv,
            dd_penalty=args.dd_penalty,
            min_improvement=args.min_improvement,
        )
        print_proc(proc)

        row = read_last_row(results_tsv) or {}
        suggestion = row.get("suggestion", "crash")
        score = row.get("score", "")

        if suggestion == "keep":
            base_text = mutated_text
            param_weights[param] *= args.weight_up
            print(f"[{i:03d}] KEEP (score={score or 'n/a'})")
        elif suggestion == "discard":
            strategy_file.write_text(base_text, encoding="utf-8")
            param_weights[param] *= args.weight_down
            print(f"[{i:03d}] DISCARD -> reverted change")
        else:
            strategy_file.write_text(base_text, encoding="utf-8")
            param_weights[param] *= args.weight_crash
            print(f"[{i:03d}] {suggestion.upper()} -> reverted change")

        # Clamp weights to avoid collapse/explosion.
        for name in param_weights:
            param_weights[name] = min(max(param_weights[name], 0.05), 20.0)

        review_record = {
            "iteration": i,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "param": param,
            "from": current,
            "to": proposed,
            "suggestion": suggestion,
            "status": row.get("status"),
            "score": to_float(score),
            "param_weight": param_weights[param],
            "description": desc,
        }
        with review_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(review_record) + "\n")

    print(f"Autoresearch loop finished. Review log: {review_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
