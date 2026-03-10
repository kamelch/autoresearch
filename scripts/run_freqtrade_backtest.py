#!/usr/bin/env python3
"""Run a Freqtrade backtest experiment and log comparable metrics.

This script is designed for an autoresearch loop:
- executes `freqtrade backtesting`
- extracts summary metrics from exported JSON (or backtest log fallback)
- computes a scalar score
- appends a row to a TSV log
- emits keep/discard/crash suggestion based on previous keeps
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import re
import shlex
import subprocess
import sys
from typing import Any


HEADER = [
    "timestamp",
    "commit",
    "profit_pct",
    "max_drawdown_pct",
    "sharpe",
    "trades",
    "score",
    "status",
    "suggestion",
    "description",
    "log_path",
    "json_path",
]


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, str):
        txt = value.strip().replace("%", "")
        if txt == "":
            return None
        try:
            v = float(txt)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def pick_first(d: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in d:
            val = to_float(d.get(key))
            if val is not None:
                return val
    return None


def parse_metrics_from_result_json(path: pathlib.Path, strategy: str) -> dict[str, float | None]:
    if not path.exists():
        return {"profit_pct": None, "max_drawdown_pct": None, "sharpe": None, "trades": None}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    row: dict[str, Any] | None = None

    # Most stable location in many freqtrade versions.
    if isinstance(data, dict) and isinstance(data.get("strategy_comparison"), list):
        rows = data["strategy_comparison"]
        if rows:
            for candidate in rows:
                if not isinstance(candidate, dict):
                    continue
                name = candidate.get("key") or candidate.get("strategy_name") or candidate.get("strategy")
                if name == strategy:
                    row = candidate
                    break
            if row is None and isinstance(rows[0], dict):
                row = rows[0]

    # Fallback to strategy summary area.
    if row is None and isinstance(data, dict) and isinstance(data.get("strategy"), dict):
        strategy_map = data["strategy"]
        if strategy in strategy_map and isinstance(strategy_map[strategy], dict):
            row = strategy_map[strategy]
        else:
            first = next(iter(strategy_map.values()), None)
            if isinstance(first, dict):
                row = first

    if row is None:
        return {"profit_pct": None, "max_drawdown_pct": None, "sharpe": None, "trades": None}

    profit_pct = pick_first(
        row,
        [
            "profit_total_pct",
            "total_profit_pct",
            "profit_mean_pct",
            "profit_pct",
        ],
    )

    # `max_drawdown_account` is usually fraction in [0,1]. Convert to percent if needed.
    drawdown = pick_first(
        row,
        [
            "max_drawdown_pct",
            "max_drawdown_account",
            "max_drawdown",
            "max_relative_drawdown",
            "max_pct_underwater",
        ],
    )
    if drawdown is not None and 0 <= drawdown <= 1:
        drawdown = drawdown * 100.0

    sharpe = pick_first(row, ["sharpe", "sharpe_ratio"])

    trades_val = pick_first(
        row,
        [
            "trades",
            "total_trades",
            "trade_count",
        ],
    )

    return {
        "profit_pct": profit_pct,
        "max_drawdown_pct": drawdown,
        "sharpe": sharpe,
        "trades": trades_val,
    }


def parse_metrics_from_log(text: str) -> dict[str, float | None]:
    def find(patterns: list[str]) -> float | None:
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                val = to_float(m.group(1))
                if val is not None:
                    return val
        return None

    profit_pct = find(
        [
            r"Total\s+profit\s*%\s*[|:]\s*([-+]?\d+(?:\.\d+)?)",
            r"profit_total_pct\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
        ]
    )
    drawdown = find(
        [
            r"Max\s*%\s*of\s*account\s*underwater\s*[|:]\s*([-+]?\d+(?:\.\d+)?)",
            r"max_drawdown(?:_pct|_account)?\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
        ]
    )
    sharpe = find(
        [
            r"Sharpe\s*[|:]\s*([-+]?\d+(?:\.\d+)?)",
            r"sharpe(?:_ratio)?\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
        ]
    )
    trades = find(
        [
            r"Total\/Daily\s+Avg\s+Trades\s*[|:]\s*(\d+)",
            r"total_trades\s*[=:]\s*(\d+)",
        ]
    )

    if drawdown is not None and 0 <= drawdown <= 1:
        drawdown = drawdown * 100.0

    return {
        "profit_pct": profit_pct,
        "max_drawdown_pct": drawdown,
        "sharpe": sharpe,
        "trades": trades,
    }


def compute_score(profit_pct: float | None, drawdown_pct: float | None, dd_penalty: float) -> float | None:
    if profit_pct is None:
        return None
    dd = drawdown_pct if drawdown_pct is not None else 0.0
    return profit_pct - dd_penalty * dd


def read_best_keep_score(tsv_path: pathlib.Path) -> float | None:
    if not tsv_path.exists():
        return None

    best = None
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep":
                continue
            score = to_float(row.get("score"))
            if score is None:
                continue
            if best is None or score > best:
                best = score
    return best


def ensure_tsv_header(tsv_path: pathlib.Path) -> None:
    if tsv_path.exists() and tsv_path.stat().st_size > 0:
        return
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(HEADER)


def append_tsv(tsv_path: pathlib.Path, row: dict[str, Any]) -> None:
    ensure_tsv_header(tsv_path)
    with tsv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([row.get(col, "") for col in HEADER])


def git_short_commit(repo_dir: pathlib.Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def rel_or_abs(path: pathlib.Path, base: pathlib.Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Freqtrade backtest experiment and log metrics")
    parser.add_argument("--freqtrade-dir", required=True, help="Path to your Freqtrade project root")
    parser.add_argument("--config", required=True, help="Config path passed to freqtrade")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--strategy-path", default=None, help="Strategy path passed to freqtrade")
    parser.add_argument("--freqaimodel", default=None, help="FreqAI model class name")
    parser.add_argument("--timerange", required=True, help="Backtest timerange, e.g. 20250101-20260301")
    parser.add_argument("--description", default="", help="Short description for TSV log")
    parser.add_argument("--repo-dir", default=None, help="Git repo for commit hash + tsv default")
    parser.add_argument("--results-tsv", default=None, help="TSV output path")
    parser.add_argument("--runs-dir", default=None, help="Directory for logs/json artifacts")
    parser.add_argument("--dd-penalty", type=float, default=0.5, help="Score = profit_pct - dd_penalty * drawdown_pct")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Required score delta vs best keep")
    args = parser.parse_args()

    freqtrade_dir = pathlib.Path(args.freqtrade_dir).expanduser().resolve()
    if not freqtrade_dir.exists():
        raise FileNotFoundError(f"freqtrade-dir not found: {freqtrade_dir}")

    script_repo_dir = pathlib.Path(__file__).resolve().parents[1]
    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve() if args.repo_dir else script_repo_dir

    runs_dir = pathlib.Path(args.runs_dir).expanduser().resolve() if args.runs_dir else (repo_dir / "freqtrade" / "runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    results_tsv = pathlib.Path(args.results_tsv).expanduser().resolve() if args.results_tsv else (repo_dir / "freqtrade" / "results.tsv")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.strategy)
    backtest_json = runs_dir / f"backtest_{safe_name}_{stamp}.json"
    run_log = runs_dir / f"run_{safe_name}_{stamp}.log"

    cmd = [
        "freqtrade",
        "backtesting",
        "--config",
        args.config,
        "--strategy",
        args.strategy,
        "--timerange",
        args.timerange,
        "--export",
        "trades",
        "--backtest-filename",
        str(backtest_json),
    ]
    if args.strategy_path:
        cmd.extend(["--strategy-path", args.strategy_path])
    if args.freqaimodel:
        cmd.extend(["--freqaimodel", args.freqaimodel])

    proc = subprocess.run(
        cmd,
        cwd=freqtrade_dir,
        text=True,
        capture_output=True,
    )

    log_text = f"$ {shlex.join(cmd)}\n\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n"
    run_log.write_text(log_text, encoding="utf-8")

    metrics = parse_metrics_from_result_json(backtest_json, args.strategy)
    if metrics["profit_pct"] is None:
        fallback = parse_metrics_from_log(log_text)
        for key, value in fallback.items():
            if metrics.get(key) is None:
                metrics[key] = value

    score = compute_score(metrics["profit_pct"], metrics["max_drawdown_pct"], args.dd_penalty)

    status = "crash" if proc.returncode != 0 else "ok"
    suggestion = "crash"

    if status == "ok" and score is not None:
        best_keep = read_best_keep_score(results_tsv)
        if best_keep is None or score > best_keep + args.min_improvement:
            suggestion = "keep"
            status = "keep"
        else:
            suggestion = "discard"
            status = "discard"

    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    row = {
        "timestamp": timestamp,
        "commit": git_short_commit(repo_dir),
        "profit_pct": f"{metrics['profit_pct']:.6f}" if metrics["profit_pct"] is not None else "",
        "max_drawdown_pct": f"{metrics['max_drawdown_pct']:.6f}" if metrics["max_drawdown_pct"] is not None else "",
        "sharpe": f"{metrics['sharpe']:.6f}" if metrics["sharpe"] is not None else "",
        "trades": f"{int(metrics['trades'])}" if metrics["trades"] is not None else "",
        "score": f"{score:.6f}" if score is not None else "",
        "status": status,
        "suggestion": suggestion,
        "description": args.description,
        "log_path": rel_or_abs(run_log, repo_dir),
        "json_path": rel_or_abs(backtest_json, repo_dir),
    }
    append_tsv(results_tsv, row)

    print("---")
    print(f"status:            {status}")
    print(f"suggestion:        {suggestion}")
    print(f"profit_pct:        {row['profit_pct'] or 'n/a'}")
    print(f"max_drawdown_pct:  {row['max_drawdown_pct'] or 'n/a'}")
    print(f"sharpe:            {row['sharpe'] or 'n/a'}")
    print(f"trades:            {row['trades'] or 'n/a'}")
    print(f"score:             {row['score'] or 'n/a'}")
    print(f"results_tsv:       {results_tsv}")
    print(f"run_log:           {run_log}")
    print(f"backtest_json:     {backtest_json}")

    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
