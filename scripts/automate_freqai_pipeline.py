#!/usr/bin/env python3
"""Automate FreqAI integration pipeline with campaign-isolated train/holdout flow."""

from __future__ import annotations

import argparse
import csv
import pathlib
import shlex
import shutil
import subprocess
import sys
from typing import Any

from autoresearch_utils import (
    build_campaign_id,
    config_fingerprint,
    extract_pair_whitelist,
    load_config_json,
    pairlist_fingerprint,
    resolve_config_path,
)


def run_cmd(cmd: list[str], cwd: pathlib.Path) -> int:
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


def resolve_freqtrade_cmd(freqtrade_dir: pathlib.Path, freqtrade_bin: str | None) -> list[str]:
    if freqtrade_bin:
        tokens = shlex.split(freqtrade_bin)
        if len(tokens) == 1:
            candidate = pathlib.Path(tokens[0]).expanduser()
            if candidate.exists():
                return [str(candidate.resolve())]
            hit = shutil.which(tokens[0])
            if hit:
                return [hit]
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


def summarize_results(tsv_path: pathlib.Path, campaign_id: str, top_n: int = 5) -> None:
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        print(f"No results file found at {tsv_path}")
        return

    rows: list[dict[str, Any]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("campaign_id") != campaign_id:
                continue
            if row.get("stage") != "holdout":
                continue
            score = to_float(row.get("score"))
            if score is None:
                continue
            row["_score"] = score
            rows.append(row)

    if not rows:
        print("No scored holdout rows found yet for this campaign.")
        return

    rows.sort(key=lambda r: float(r["_score"]), reverse=True)
    print("---")
    print(f"Top {min(top_n, len(rows))} holdout experiments by score for campaign {campaign_id}:")
    for idx, row in enumerate(rows[:top_n], start=1):
        print(
            f"{idx:02d}. score={row.get('score')} status={row.get('status')} "
            f"suggestion={row.get('suggestion')} reason={row.get('decision_reason')} desc={row.get('description')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automate FreqAI setup + train/holdout autoresearch loop")
    parser.add_argument("--freqtrade-dir", required=True, help="Path to Freqtrade project root")
    parser.add_argument("--freqtrade-bin", default=None, help="Freqtrade executable (or command), e.g. /path/.venv/bin/freqtrade")
    parser.add_argument("--train-timerange", required=True, help="Optimization timerange")
    parser.add_argument("--holdout-timerange", required=True, help="Holdout timerange")
    parser.add_argument("--iterations", type=int, default=30, help="Autoresearch iterations")
    parser.add_argument("--strategy", default="AutoresearchFreqAIStrategy", help="Strategy class name")
    parser.add_argument("--freqaimodel", default="AutoresearchLSTMRegressor", help="FreqAI model")
    parser.add_argument("--freqaimodel-path", default=None, help="Lookup path for custom FreqAI models")
    parser.add_argument("--campaign-id", default=None, help="Campaign ID (default: deterministic hash)")
    parser.add_argument("--config-dest", default="user_data/config_freqai_autoresearch.json", help="Config path relative to freqtrade-dir")
    parser.add_argument("--download-timeframes", default="5m 15m 1h", help="Space-separated timeframes for download-data")
    parser.add_argument("--download-timerange", default="20240101-20260301", help="Timerange for download-data")
    parser.add_argument("--skip-download", action="store_true", help="Skip freqtrade download-data step")
    parser.add_argument("--overwrite-config", action="store_true", help="Overwrite existing config-dest if present")
    parser.add_argument("--repo-dir", default=None, help="Path to this repo (default: auto)")
    parser.add_argument("--strategy-path", default=None, help="Strategy path for freqtrade (default: repo/freqtrade/strategies)")
    parser.add_argument(
        "--profile",
        default=None,
        help="Mutation profile path (relative to repo unless absolute). If omitted, prefers campaign auto-profile when available.",
    )
    parser.add_argument("--auto-profile", default=None, help="Adaptive profile output path (default: campaign-scoped)")
    parser.add_argument("--space-state", default=None, help="Adaptive space-state file path (default: campaign-scoped)")
    parser.add_argument("--results-tsv", default=None, help="Results TSV path (default: campaign-scoped)")
    parser.add_argument("--no-space-adaptation", action="store_true", help="Disable automatic search-space adaptation")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Required score improvement to keep")
    parser.add_argument("--dd-penalty", type=float, default=0.5, help="Score penalty weight on drawdown")

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
    if not freqtrade_dir.exists():
        raise FileNotFoundError(f"freqtrade-dir not found: {freqtrade_dir}")
    freqtrade_cmd = resolve_freqtrade_cmd(freqtrade_dir, args.freqtrade_bin)

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

    config_path = resolve_config_path(freqtrade_dir, args.config_dest)
    config_data = load_config_json(config_path)
    cfg_fp = config_fingerprint(config_data)
    pairs = extract_pair_whitelist(config_data)
    pair_fp = pairlist_fingerprint(pairs)

    campaign_id = args.campaign_id or build_campaign_id(
        strategy=args.strategy,
        freqaimodel=args.freqaimodel,
        train_timerange=args.train_timerange,
        holdout_timerange=args.holdout_timerange,
        config_fp=cfg_fp,
        pair_fp=pair_fp,
    )

    strategy_path = args.strategy_path or str((repo_dir / "freqtrade" / "strategies").resolve())
    freqaimodel_path = args.freqaimodel_path or str((repo_dir / "freqtrade" / "freqaimodels").resolve())

    auto_profile_path = pathlib.Path(args.auto_profile).expanduser().resolve() if args.auto_profile else (
        repo_dir / "freqtrade" / f"autoresearch_profile.{campaign_id}.json"
    )

    if args.profile:
        profile_path = pathlib.Path(args.profile).expanduser()
        if not profile_path.is_absolute():
            profile_path = (repo_dir / profile_path).resolve()
    else:
        default_profile = (repo_dir / "freqtrade" / "autoresearch_profile.example.json").resolve()
        profile_path = auto_profile_path if auto_profile_path.exists() else default_profile

    results_tsv = pathlib.Path(args.results_tsv).expanduser().resolve() if args.results_tsv else (
        repo_dir / "freqtrade" / "results" / f"results_{campaign_id}.tsv"
    )

    space_state_path = pathlib.Path(args.space_state).expanduser().resolve() if args.space_state else (
        repo_dir / "freqtrade" / "runs" / f"space_state_{args.strategy}_{campaign_id}.json"
    )

    print(f"Campaign ID: {campaign_id}")
    print(f"Using profile: {profile_path}")
    print(f"Auto profile output: {auto_profile_path}")
    print(f"Results TSV: {results_tsv}")
    print(f"Space state: {space_state_path}")
    print(f"FreqAI model path: {freqaimodel_path}")

    if not args.skip_download:
        download_cmd = freqtrade_cmd + [
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

    baseline_candidate = "baseline"
    baseline_train_cmd = [
        sys.executable,
        str(repo_dir / "scripts" / "run_freqtrade_backtest.py"),
        "--freqtrade-dir",
        str(freqtrade_dir),
        "--freqtrade-bin",
        shlex.join(freqtrade_cmd),
        "--config",
        args.config_dest,
        "--strategy",
        args.strategy,
        "--strategy-path",
        strategy_path,
        "--freqaimodel",
        args.freqaimodel,
        "--freqaimodel-path",
        freqaimodel_path,
        "--timerange",
        args.train_timerange,
        "--description",
        "baseline-auto [train]",
        "--repo-dir",
        str(repo_dir),
        "--results-tsv",
        str(results_tsv),
        "--dd-penalty",
        str(args.dd_penalty),
        "--min-improvement",
        str(args.min_improvement),
        "--campaign-id",
        campaign_id,
        "--candidate-id",
        baseline_candidate,
        "--stage",
        "train",
        "--min-profit-pct",
        str(args.min_profit_pct),
        "--max-drawdown-pct",
        str(args.max_drawdown_pct),
        "--min-sharpe",
        str(args.min_sharpe),
        "--pair-min-trades-floor",
        str(args.pair_min_trades_floor),
        "--pair-min-trades-mode",
        args.pair_min_trades_mode,
    ]
    rc = run_cmd(baseline_train_cmd, cwd=repo_dir)
    if rc != 0:
        return rc

    baseline_train_row = read_last_matching_row(results_tsv, campaign_id, baseline_candidate, "train") or {}
    if baseline_train_row.get("suggestion") == "keep":
        baseline_holdout_cmd = [
            sys.executable,
            str(repo_dir / "scripts" / "run_freqtrade_backtest.py"),
            "--freqtrade-dir",
            str(freqtrade_dir),
            "--freqtrade-bin",
            shlex.join(freqtrade_cmd),
            "--config",
            args.config_dest,
            "--strategy",
            args.strategy,
            "--strategy-path",
            strategy_path,
            "--freqaimodel",
            args.freqaimodel,
            "--freqaimodel-path",
            freqaimodel_path,
            "--timerange",
            args.holdout_timerange,
            "--description",
            "baseline-auto [holdout]",
            "--repo-dir",
            str(repo_dir),
            "--results-tsv",
            str(results_tsv),
            "--dd-penalty",
            str(args.dd_penalty),
            "--min-improvement",
            str(args.min_improvement),
            "--campaign-id",
            campaign_id,
            "--candidate-id",
            baseline_candidate,
            "--stage",
            "holdout",
            "--min-profit-pct",
            str(args.min_profit_pct),
            "--max-drawdown-pct",
            str(args.max_drawdown_pct),
            "--min-sharpe",
            str(args.min_sharpe),
            "--pair-min-trades-floor",
            str(args.pair_min_trades_floor),
            "--pair-min-trades-mode",
            args.pair_min_trades_mode,
        ]
        rc = run_cmd(baseline_holdout_cmd, cwd=repo_dir)
        if rc != 0:
            return rc
    else:
        print("Baseline train stage did not pass gates; holdout baseline skipped.")

    loop_cmd = [
        sys.executable,
        str(repo_dir / "scripts" / "freqai_autoresearch_loop.py"),
        "--freqtrade-dir",
        str(freqtrade_dir),
        "--freqtrade-bin",
        shlex.join(freqtrade_cmd),
        "--config",
        args.config_dest,
        "--strategy",
        args.strategy,
        "--strategy-path",
        strategy_path,
        "--freqaimodel",
        args.freqaimodel,
        "--freqaimodel-path",
        freqaimodel_path,
        "--train-timerange",
        args.train_timerange,
        "--holdout-timerange",
        args.holdout_timerange,
        "--iterations",
        str(args.iterations),
        "--profile",
        str(profile_path),
        "--save-profile",
        str(auto_profile_path),
        "--space-state",
        str(space_state_path),
        "--campaign-id",
        campaign_id,
        "--repo-dir",
        str(repo_dir),
        "--results-tsv",
        str(results_tsv),
        "--dd-penalty",
        str(args.dd_penalty),
        "--min-improvement",
        str(args.min_improvement),
        "--min-profit-pct",
        str(args.min_profit_pct),
        "--max-drawdown-pct",
        str(args.max_drawdown_pct),
        "--min-sharpe",
        str(args.min_sharpe),
        "--pair-min-trades-floor",
        str(args.pair_min_trades_floor),
        "--pair-min-trades-mode",
        args.pair_min_trades_mode,
    ]
    if args.no_space_adaptation:
        loop_cmd.append("--no-space-adaptation")

    rc = run_cmd(loop_cmd, cwd=repo_dir)
    if rc != 0:
        return rc

    summarize_results(results_tsv, campaign_id=campaign_id, top_n=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
