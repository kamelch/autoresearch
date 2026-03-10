#!/usr/bin/env python3
"""Automate FreqAI integration pipeline (steps 1-6) end-to-end.

Pipeline:
1) install/update config in Freqtrade user_data
2) download data
3) baseline backtest + metric log
4) iterative autoresearch loop with adaptive review
5) print top results summary
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import shutil
import subprocess
import sys
from typing import Any


def run_cmd(cmd: list[str], cwd: pathlib.Path) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print("[stderr]")
        print(proc.stderr.rstrip())
    return proc.returncode


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


def summarize_results(tsv_path: pathlib.Path, top_n: int = 5) -> None:
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        print(f"No results file found at {tsv_path}")
        return

    rows: list[dict[str, Any]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            score = to_float(row.get("score"))
            if score is None:
                continue
            row["_score"] = score
            rows.append(row)

    if not rows:
        print("No scored rows found yet.")
        return

    rows.sort(key=lambda r: float(r["_score"]), reverse=True)
    print("---")
    print(f"Top {min(top_n, len(rows))} experiments by score:")
    for idx, row in enumerate(rows[:top_n], start=1):
        print(
            f"{idx:02d}. score={row.get('score')} status={row.get('status')} "
            f"suggestion={row.get('suggestion')} desc={row.get('description')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Automate FreqAI setup + backtest + autoresearch loop")
    parser.add_argument("--freqtrade-dir", required=True, help="Path to Freqtrade project root")
    parser.add_argument("--timerange", required=True, help="Backtest timerange used for optimization")
    parser.add_argument("--iterations", type=int, default=30, help="Autoresearch iterations")
    parser.add_argument("--strategy", default="AutoresearchFreqAIStrategy", help="Strategy class name")
    parser.add_argument("--freqaimodel", default="PyTorchMLPRegressor", help="FreqAI model")
    parser.add_argument("--config-dest", default="user_data/config_freqai_autoresearch.json", help="Config path relative to freqtrade-dir")
    parser.add_argument("--download-timeframes", default="5m 15m 1h", help="Space-separated timeframes for download-data")
    parser.add_argument("--download-timerange", default="20240101-20260301", help="Timerange for download-data")
    parser.add_argument("--skip-download", action="store_true", help="Skip freqtrade download-data step")
    parser.add_argument("--overwrite-config", action="store_true", help="Overwrite existing config-dest if present")
    parser.add_argument("--repo-dir", default=None, help="Path to this repo (default: auto)")
    parser.add_argument("--strategy-path", default=None, help="Strategy path for freqtrade (default: repo/freqtrade/strategies)")
    parser.add_argument("--profile", default="freqtrade/autoresearch_profile.example.json", help="Mutation profile path (relative to repo unless absolute)")
    parser.add_argument("--results-tsv", default=None, help="Results TSV path (default: repo/freqtrade/results.tsv)")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Required score improvement to keep")
    parser.add_argument("--dd-penalty", type=float, default=0.5, help="Score penalty weight on drawdown")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve() if args.repo_dir else pathlib.Path(__file__).resolve().parents[1]
    freqtrade_dir = pathlib.Path(args.freqtrade_dir).expanduser().resolve()
    if not freqtrade_dir.exists():
        raise FileNotFoundError(f"freqtrade-dir not found: {freqtrade_dir}")

    config_template = repo_dir / "freqtrade" / "config" / "config_freqai_autoresearch.example.json"
    if not config_template.exists():
        raise FileNotFoundError(f"Config template not found: {config_template}")

    config_dest = freqtrade_dir / args.config_dest
    config_dest.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_config or not config_dest.exists():
        shutil.copy2(config_template, config_dest)
        print(f"Config installed: {config_dest}")
    else:
        print(f"Config already exists, keeping: {config_dest}")

    strategy_path = args.strategy_path or str((repo_dir / "freqtrade" / "strategies").resolve())
    profile_path = pathlib.Path(args.profile).expanduser()
    if not profile_path.is_absolute():
        profile_path = (repo_dir / profile_path).resolve()

    results_tsv = pathlib.Path(args.results_tsv).expanduser().resolve() if args.results_tsv else (repo_dir / "freqtrade" / "results.tsv")

    # Step 2: download data
    if not args.skip_download:
        download_cmd = [
            "freqtrade",
            "download-data",
            "--config",
            args.config_dest,
            "--timeframes",
            *args.download_timeframes.split(),
            "--timerange",
            args.download_timerange,
        ]
        rc = run_cmd(download_cmd, cwd=freqtrade_dir)
        if rc != 0:
            return rc

    # Step 3: baseline run
    baseline_cmd = [
        sys.executable,
        str(repo_dir / "scripts" / "run_freqtrade_backtest.py"),
        "--freqtrade-dir",
        str(freqtrade_dir),
        "--config",
        args.config_dest,
        "--strategy",
        args.strategy,
        "--strategy-path",
        strategy_path,
        "--freqaimodel",
        args.freqaimodel,
        "--timerange",
        args.timerange,
        "--description",
        "baseline-auto",
        "--repo-dir",
        str(repo_dir),
        "--results-tsv",
        str(results_tsv),
        "--dd-penalty",
        str(args.dd_penalty),
        "--min-improvement",
        str(args.min_improvement),
    ]
    rc = run_cmd(baseline_cmd, cwd=repo_dir)
    if rc != 0:
        return rc

    # Step 4: iterative loop
    loop_cmd = [
        sys.executable,
        str(repo_dir / "scripts" / "freqai_autoresearch_loop.py"),
        "--freqtrade-dir",
        str(freqtrade_dir),
        "--config",
        args.config_dest,
        "--strategy",
        args.strategy,
        "--strategy-path",
        strategy_path,
        "--freqaimodel",
        args.freqaimodel,
        "--timerange",
        args.timerange,
        "--iterations",
        str(args.iterations),
        "--profile",
        str(profile_path),
        "--repo-dir",
        str(repo_dir),
        "--results-tsv",
        str(results_tsv),
        "--dd-penalty",
        str(args.dd_penalty),
        "--min-improvement",
        str(args.min_improvement),
    ]
    rc = run_cmd(loop_cmd, cwd=repo_dir)
    if rc != 0:
        return rc

    # Step 5: summary review
    summarize_results(results_tsv, top_n=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
