#!/usr/bin/env python3
"""Run one Freqtrade backtest experiment and log campaign-scoped metrics.

v2 behavior:
- campaign-aware result rows (train/holdout + candidate IDs)
- robust artifact parsing (.json/.meta.json/.zip)
- hard risk gates before relative score comparison
- pair-coverage gate that scales with pairlist length
- keep/discard decisions isolated per campaign (holdout only)
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
import shutil
import subprocess
import sys
import zipfile
from typing import Any

try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None  # type: ignore[assignment]

from autoresearch_utils import (
    config_fingerprint,
    extract_pair_whitelist,
    load_config_json,
    pairlist_fingerprint,
    resolve_config_path,
)


HEADER = [
    "timestamp",
    "commit",
    "campaign_id",
    "candidate_id",
    "stage",
    "timerange",
    "config_fingerprint",
    "pairlist_fingerprint",
    "profit_pct",
    "max_drawdown_pct",
    "sharpe",
    "trades",
    "score",
    "status",
    "suggestion",
    "decision_reason",
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


def _load_backtest_dict(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    # Freqtrade commonly writes .meta.json + .zip.
    if str(path).endswith(".meta.json"):
        zip_candidate = pathlib.Path(str(path).replace(".meta.json", ".zip"))
        if zip_candidate.exists():
            path = zip_candidate
        else:
            json_candidate = pathlib.Path(str(path).replace(".meta.json", ".json"))
            if json_candidate.exists():
                path = json_candidate

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            json_names = [
                name
                for name in zf.namelist()
                if name.endswith(".json") and not name.endswith("_config.json")
            ]
            if not json_names:
                return None
            json_name = sorted(json_names)[0]
            data = json.loads(zf.read(json_name))
            return data if isinstance(data, dict) else None

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    return None


def extract_strategy_payload(data: dict[str, Any], strategy: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (strategy_comparison_row, strategy_blob)."""
    comparison_row: dict[str, Any] | None = None
    strategy_blob: dict[str, Any] | None = None

    if isinstance(data.get("strategy_comparison"), list):
        rows = data["strategy_comparison"]
        for candidate in rows:
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("key") or candidate.get("strategy_name") or candidate.get("strategy")
            if name == strategy:
                comparison_row = candidate
                break
        if comparison_row is None:
            for candidate in rows:
                if isinstance(candidate, dict):
                    comparison_row = candidate
                    break

    strategy_map = data.get("strategy")
    if isinstance(strategy_map, dict):
        if strategy in strategy_map and isinstance(strategy_map[strategy], dict):
            strategy_blob = strategy_map[strategy]
        else:
            first = next(iter(strategy_map.values()), None)
            if isinstance(first, dict):
                strategy_blob = first

    return comparison_row, strategy_blob


def parse_metrics_from_result_json(path: pathlib.Path, strategy: str) -> dict[str, Any]:
    data = _load_backtest_dict(path)
    out: dict[str, Any] = {
        "profit_pct": None,
        "max_drawdown_pct": None,
        "sharpe": None,
        "trades": None,
        "results_per_pair": [],
        "exit_reason_summary": [],
        "pairlist": [],
    }
    if not isinstance(data, dict):
        return out

    row, strategy_blob = extract_strategy_payload(data, strategy)

    if row is None and strategy_blob is None:
        return out

    metrics_row = row if row is not None else strategy_blob
    assert metrics_row is not None

    profit_pct = pick_first(
        metrics_row,
        [
            "profit_total_pct",
            "total_profit_pct",
            "profit_mean_pct",
            "profit_pct",
        ],
    )

    drawdown = pick_first(
        metrics_row,
        [
            "max_drawdown_pct",
            "max_drawdown_account",
            "max_drawdown",
            "max_relative_drawdown",
            "max_pct_underwater",
        ],
    )
    if drawdown is not None and 0 <= drawdown <= 1:
        drawdown *= 100.0

    sharpe = pick_first(metrics_row, ["sharpe", "sharpe_ratio"])

    trades_val = pick_first(
        metrics_row,
        [
            "trades",
            "total_trades",
            "trade_count",
        ],
    )

    if strategy_blob is not None:
        if isinstance(strategy_blob.get("results_per_pair"), list):
            out["results_per_pair"] = [r for r in strategy_blob["results_per_pair"] if isinstance(r, dict)]
        if isinstance(strategy_blob.get("exit_reason_summary"), list):
            out["exit_reason_summary"] = [r for r in strategy_blob["exit_reason_summary"] if isinstance(r, dict)]
        if isinstance(strategy_blob.get("pairlist"), list):
            out["pairlist"] = [p for p in strategy_blob["pairlist"] if isinstance(p, str) and p.strip()]

        # Prefer strategy-level metrics if comparison row missed anything.
        if profit_pct is None:
            profit_pct = pick_first(strategy_blob, ["profit_total_pct", "profit_total", "total_profit_pct"])  # type: ignore[arg-type]
            if profit_pct is not None and abs(profit_pct) <= 1.0 and "profit_total_pct" not in strategy_blob:
                profit_pct *= 100.0
        if drawdown is None:
            drawdown = pick_first(strategy_blob, ["max_drawdown_account", "max_drawdown_pct", "max_pct_underwater"])
            if drawdown is not None and 0 <= drawdown <= 1:
                drawdown *= 100.0
        if sharpe is None:
            sharpe = pick_first(strategy_blob, ["sharpe", "sharpe_ratio"])
        if trades_val is None:
            trades_val = pick_first(strategy_blob, ["total_trades", "trades", "trade_count"])

    out.update(
        {
            "profit_pct": profit_pct,
            "max_drawdown_pct": drawdown,
            "sharpe": sharpe,
            "trades": trades_val,
        }
    )
    return out


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
            r"Tot\s+Profit\s*%\s*[|:]\s*([-+]?\d+(?:\.\d+)?)",
            r"profit_total_pct\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
        ]
    )
    drawdown = find(
        [
            r"Max\s*%\s*of\s*account\s*underwater\s*[|:]\s*([-+]?\d+(?:\.\d+)?)",
            r"Drawdown[^\n]*?([0-9]+(?:\.\d+)?)%",
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
            r"Total/Daily\s+Avg\s+Trades\s*[|:]\s*(\d+)",
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


def parse_strategy_row_metrics_from_log(text: str, strategy: str) -> dict[str, float | None]:
    row_re = re.compile(
        rf"[│|]\s*{re.escape(strategy)}\s*[│|]\s*(\d+)\s*[│|]\s*([-+]?\d+(?:\.\d+)?)\s*[│|]\s*"
        rf"([-+]?\d+(?:\.\d+)?)\s*[│|]\s*([-+]?\d+(?:\.\d+)?)\s*[│|][^\n]*?([0-9]+(?:\.\d+)?)%\s*[│|]"
    )
    m = row_re.search(text)
    if not m:
        return {"profit_pct": None, "max_drawdown_pct": None, "sharpe": None, "trades": None}

    trades = to_float(m.group(1))
    profit_pct = to_float(m.group(4))
    drawdown_pct = to_float(m.group(5))
    return {
        "profit_pct": profit_pct,
        "max_drawdown_pct": drawdown_pct,
        "sharpe": None,
        "trades": trades,
    }


def discover_backtest_artifact(log_text: str, freqtrade_dir: pathlib.Path, preferred: pathlib.Path) -> pathlib.Path:
    if preferred.exists():
        return preferred

    m = re.search(r'dumping json to "([^"]+)"', log_text)
    if m:
        dumped = pathlib.Path(m.group(1))
        if dumped.exists():
            return dumped
        if str(dumped).endswith(".meta.json"):
            zip_candidate = pathlib.Path(str(dumped).replace(".meta.json", ".zip"))
            if zip_candidate.exists():
                return zip_candidate
            json_candidate = pathlib.Path(str(dumped).replace(".meta.json", ".json"))
            if json_candidate.exists():
                return json_candidate

    bt_dir = freqtrade_dir / "user_data" / "backtest_results"
    if bt_dir.exists():
        zips = sorted(bt_dir.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if zips:
            return zips[0]

    return preferred


def compute_score(profit_pct: float | None, drawdown_pct: float | None, dd_penalty: float) -> float | None:
    if profit_pct is None:
        return None
    dd = drawdown_pct if drawdown_pct is not None else 0.0
    return profit_pct - dd_penalty * dd


def normalize_pair_trade_map(results_per_pair: list[dict[str, Any]], pairlist: list[str]) -> dict[str, int]:
    pair_trades: dict[str, int] = {}
    for row in results_per_pair:
        key = row.get("key")
        if not isinstance(key, str):
            continue
        if key.strip().upper() == "TOTAL":
            continue
        trades = to_float(row.get("trades"))
        pair_trades[key] = int(trades or 0)

    # Ensure all configured pairs are represented, even if they got 0 trades.
    for pair in pairlist:
        pair_trades.setdefault(pair, 0)

    return pair_trades


def evaluate_pair_coverage(
    total_trades: float | None,
    results_per_pair: list[dict[str, Any]],
    pairlist: list[str],
    pair_min_trades_floor: int,
    pair_min_trades_mode: str,
) -> dict[str, Any]:
    pair_trade_map = normalize_pair_trade_map(results_per_pair, pairlist)
    num_pairs = len(pair_trade_map)

    if pair_min_trades_mode != "dynamic":
        return {
            "ok": True,
            "reason": "pair_coverage_disabled",
            "threshold": 0,
            "pair_trade_map": pair_trade_map,
            "missing_pairs": [],
        }

    if num_pairs == 0:
        return {
            "ok": False,
            "reason": "pair_coverage_missing",
            "threshold": pair_min_trades_floor,
            "pair_trade_map": pair_trade_map,
            "missing_pairs": [],
        }

    trades = int(total_trades or 0)
    dynamic_threshold = int(math.ceil(trades / (2 * num_pairs))) if trades > 0 else 0
    threshold = max(int(pair_min_trades_floor), dynamic_threshold)

    missing_pairs = [pair for pair, pair_trades in pair_trade_map.items() if pair_trades < threshold]
    ok = len(missing_pairs) == 0
    reason = "ok" if ok else "pair_coverage_failed"

    return {
        "ok": ok,
        "reason": reason,
        "threshold": threshold,
        "pair_trade_map": pair_trade_map,
        "missing_pairs": missing_pairs,
    }


def evaluate_hard_gates(
    profit_pct: float,
    drawdown_pct: float,
    sharpe: float,
    pair_gate: dict[str, Any],
    min_profit_pct: float,
    max_drawdown_pct: float,
    min_sharpe: float,
) -> tuple[bool, str]:
    if profit_pct < min_profit_pct:
        return False, "profit_below_min"
    if drawdown_pct > max_drawdown_pct:
        return False, "drawdown_above_max"
    if sharpe < min_sharpe:
        return False, "sharpe_below_min"
    if not bool(pair_gate.get("ok")):
        return False, str(pair_gate.get("reason") or "pair_coverage_failed")
    return True, "hard_gates_passed"


def read_best_keep_score(tsv_path: pathlib.Path, campaign_id: str, stage: str = "holdout") -> float | None:
    if not tsv_path.exists():
        return None

    best = None
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep":
                continue
            if row.get("campaign_id") != campaign_id:
                continue
            if row.get("stage") != stage:
                continue
            score = to_float(row.get("score"))
            if score is None:
                continue
            if best is None or score > best:
                best = score
    return best


def ensure_tsv_header(tsv_path: pathlib.Path) -> None:
    if tsv_path.exists() and tsv_path.stat().st_size > 0:
        with tsv_path.open("r", encoding="utf-8") as f:
            existing_header = f.readline().rstrip("\n").split("\t")
        if existing_header != HEADER:
            raise RuntimeError(
                f"TSV header mismatch in {tsv_path}. "
                "Use a campaign-specific results file (e.g. freqtrade/results/results_<campaign_id>.tsv)."
            )
        return
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(HEADER)


def append_tsv(tsv_path: pathlib.Path, row: dict[str, Any]) -> None:
    ensure_tsv_header(tsv_path)
    lock_handle = None
    try:
        if fcntl is not None:
            lock_handle = (tsv_path.with_suffix(tsv_path.suffix + ".lock")).open("a+", encoding="utf-8")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        with tsv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([row.get(col, "") for col in HEADER])
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


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


def format_pair_counts(pair_trade_map: dict[str, int]) -> str:
    if not pair_trade_map:
        return "n/a"
    ordered = sorted(pair_trade_map.items(), key=lambda kv: kv[0])
    return ", ".join(f"{pair}={trades}" for pair, trades in ordered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Freqtrade backtest experiment and log metrics")
    parser.add_argument("--freqtrade-dir", required=True, help="Path to your Freqtrade project root")
    parser.add_argument("--config", required=True, help="Config path passed to freqtrade")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--strategy-path", default=None, help="Strategy path passed to freqtrade")
    parser.add_argument("--freqaimodel", default="AutoresearchLSTMRegressor", help="FreqAI model class name")
    parser.add_argument("--freqaimodel-path", default=None, help="Optional lookup path for custom FreqAI models")
    parser.add_argument("--freqtrade-bin", default=None, help="Freqtrade executable (or command), e.g. /path/.venv/bin/freqtrade")
    parser.add_argument("--timerange", required=True, help="Backtest timerange, e.g. 20250101-20260301")
    parser.add_argument("--description", default="", help="Short description for TSV log")
    parser.add_argument("--repo-dir", default=None, help="Git repo for commit hash + tsv default")
    parser.add_argument("--results-tsv", default=None, help="TSV output path")
    parser.add_argument("--runs-dir", default=None, help="Directory for logs/json artifacts")
    parser.add_argument("--dd-penalty", type=float, default=0.5, help="Score = profit_pct - dd_penalty * drawdown_pct")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Required score delta vs best keep")

    parser.add_argument("--campaign-id", default="default", help="Campaign identifier")
    parser.add_argument("--candidate-id", default=None, help="Candidate identifier shared by train/holdout")
    parser.add_argument("--stage", choices=["train", "holdout"], default="holdout", help="Evaluation stage")

    parser.add_argument("--min-profit-pct", type=float, default=0.0, help="Absolute gate: minimum total profit %%")
    parser.add_argument("--max-drawdown-pct", type=float, default=12.0, help="Absolute gate: maximum drawdown %%")
    parser.add_argument("--min-sharpe", type=float, default=0.0, help="Absolute gate: minimum Sharpe")
    parser.add_argument("--pair-min-trades-floor", type=int, default=10, help="Floor for pair-coverage min trades")
    parser.add_argument(
        "--pair-min-trades-mode",
        default="dynamic",
        choices=["dynamic", "off"],
        help="Pair-coverage mode: dynamic uses max(floor, ceil(total_trades/(2*num_pairs)))",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    freqtrade_dir = pathlib.Path(args.freqtrade_dir).expanduser().resolve()
    if not freqtrade_dir.exists():
        raise FileNotFoundError(f"freqtrade-dir not found: {freqtrade_dir}")

    script_repo_dir = pathlib.Path(__file__).resolve().parents[1]
    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve() if args.repo_dir else script_repo_dir

    runs_dir = pathlib.Path(args.runs_dir).expanduser().resolve() if args.runs_dir else (repo_dir / "freqtrade" / "runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    results_tsv = pathlib.Path(args.results_tsv).expanduser().resolve() if args.results_tsv else (repo_dir / "freqtrade" / "results.tsv")

    config_path = resolve_config_path(freqtrade_dir, args.config)
    config_data = load_config_json(config_path)
    config_fp = config_fingerprint(config_data)
    config_pairlist = extract_pair_whitelist(config_data)
    pair_fp = pairlist_fingerprint(config_pairlist)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.strategy)
    candidate_id = args.candidate_id or f"candidate_{stamp}"

    backtest_json = runs_dir / f"backtest_{safe_name}_{stamp}.json"
    run_log = runs_dir / f"run_{safe_name}_{stamp}.log"
    freqtrade_cmd = resolve_freqtrade_cmd(freqtrade_dir, args.freqtrade_bin)

    cmd = freqtrade_cmd + [
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
    if args.freqaimodel_path:
        cmd.extend(["--freqaimodel-path", args.freqaimodel_path])

    print(f"$ {shlex.join(cmd)}", flush=True)
    out_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=freqtrade_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line)
        print(line, end="", flush=True)
    returncode = proc.wait()
    stdout_combined = "".join(out_lines)

    log_text = f"$ {shlex.join(cmd)}\n\n{stdout_combined}\n"
    run_log.write_text(log_text, encoding="utf-8")

    artifact_path = discover_backtest_artifact(log_text, freqtrade_dir, backtest_json)
    metrics = parse_metrics_from_result_json(artifact_path, args.strategy)

    # Fill gaps with log parsing.
    fallback = parse_metrics_from_log(log_text)
    table_fallback = parse_strategy_row_metrics_from_log(log_text, args.strategy)
    for key, value in fallback.items():
        if metrics.get(key) is None:
            metrics[key] = value
    for key, value in table_fallback.items():
        if metrics.get(key) is None:
            metrics[key] = value

    # Keep pairlist from strategy output when available; otherwise config pair_whitelist.
    result_pairlist = metrics.get("pairlist") if isinstance(metrics.get("pairlist"), list) else []
    pairlist = [p for p in result_pairlist if isinstance(p, str) and p.strip()] or config_pairlist
    metrics["pairlist"] = pairlist

    score = compute_score(metrics.get("profit_pct"), metrics.get("max_drawdown_pct"), args.dd_penalty)

    status = "ok"
    suggestion = "discard"
    decision_reason = "unknown"

    profit_pct = to_float(metrics.get("profit_pct"))
    drawdown_pct = to_float(metrics.get("max_drawdown_pct"))
    sharpe = to_float(metrics.get("sharpe"))
    trades = to_float(metrics.get("trades"))

    pair_gate = evaluate_pair_coverage(
        total_trades=trades,
        results_per_pair=metrics.get("results_per_pair") if isinstance(metrics.get("results_per_pair"), list) else [],
        pairlist=pairlist,
        pair_min_trades_floor=args.pair_min_trades_floor,
        pair_min_trades_mode=args.pair_min_trades_mode,
    )

    if returncode != 0:
        status = "crash"
        suggestion = "crash"
        decision_reason = "command_failed"
    elif profit_pct is None or drawdown_pct is None or sharpe is None or trades is None:
        status = "discard"
        suggestion = "discard"
        decision_reason = "metrics_unparsable"
    elif score is None:
        status = "discard"
        suggestion = "discard"
        decision_reason = "score_unavailable"
    else:
        hard_ok, hard_reason = evaluate_hard_gates(
            profit_pct=profit_pct,
            drawdown_pct=drawdown_pct,
            sharpe=sharpe,
            pair_gate=pair_gate,
            min_profit_pct=args.min_profit_pct,
            max_drawdown_pct=args.max_drawdown_pct,
            min_sharpe=args.min_sharpe,
        )
        if not hard_ok:
            status = "discard"
            suggestion = "discard"
            decision_reason = hard_reason
        elif args.stage == "train":
            status = "keep"
            suggestion = "keep"
            decision_reason = "train_passed_hard_gates"
        else:
            best_keep = read_best_keep_score(results_tsv, campaign_id=args.campaign_id, stage="holdout")
            if best_keep is None:
                status = "keep"
                suggestion = "keep"
                decision_reason = "first_holdout_keep"
            elif score > best_keep + args.min_improvement:
                status = "keep"
                suggestion = "keep"
                decision_reason = "improved_over_best_keep"
            else:
                status = "discard"
                suggestion = "discard"
                decision_reason = "score_below_best_keep"

    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    row = {
        "timestamp": timestamp,
        "commit": git_short_commit(repo_dir),
        "campaign_id": args.campaign_id,
        "candidate_id": candidate_id,
        "stage": args.stage,
        "timerange": args.timerange,
        "config_fingerprint": config_fp,
        "pairlist_fingerprint": pair_fp,
        "profit_pct": f"{profit_pct:.6f}" if profit_pct is not None else "",
        "max_drawdown_pct": f"{drawdown_pct:.6f}" if drawdown_pct is not None else "",
        "sharpe": f"{sharpe:.6f}" if sharpe is not None else "",
        "trades": f"{int(trades)}" if trades is not None else "",
        "score": f"{score:.6f}" if score is not None else "",
        "status": status,
        "suggestion": suggestion,
        "decision_reason": decision_reason,
        "description": args.description,
        "log_path": rel_or_abs(run_log, repo_dir),
        "json_path": rel_or_abs(artifact_path, repo_dir),
    }
    append_tsv(results_tsv, row)

    print("---")
    print(f"status:            {status}")
    print(f"suggestion:        {suggestion}")
    print(f"decision_reason:   {decision_reason}")
    print(f"campaign_id:       {args.campaign_id}")
    print(f"candidate_id:      {candidate_id}")
    print(f"stage:             {args.stage}")
    print(f"profit_pct:        {row['profit_pct'] or 'n/a'}")
    print(f"max_drawdown_pct:  {row['max_drawdown_pct'] or 'n/a'}")
    print(f"sharpe:            {row['sharpe'] or 'n/a'}")
    print(f"trades:            {row['trades'] or 'n/a'}")
    print(f"score:             {row['score'] or 'n/a'}")
    print(f"pair_counts:       {format_pair_counts(pair_gate['pair_trade_map'])}")
    print(f"pair_threshold:    {pair_gate['threshold']}")
    print(f"pair_gate_ok:      {pair_gate['ok']}")
    print(f"results_tsv:       {results_tsv}")
    print(f"run_log:           {run_log}")
    print(f"backtest_json:     {artifact_path}")

    return 0 if returncode == 0 else returncode


if __name__ == "__main__":
    sys.exit(main())
