from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
import zipfile

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_freqtrade_backtest as runner


class RunnerDecisionTests(unittest.TestCase):
    def _row(self, **overrides: str) -> dict[str, str]:
        row = {key: "" for key in runner.HEADER}
        row.update(
            {
                "timestamp": "2026-03-10T00:00:00+00:00",
                "commit": "abc123",
                "campaign_id": "camp",
                "candidate_id": "cand",
                "stage": "holdout",
                "timerange": "20250101-20250301",
                "config_fingerprint": "cfg",
                "pairlist_fingerprint": "pair",
                "status": "keep",
                "suggestion": "keep",
                "decision_reason": "test",
                "description": "test-row",
            }
        )
        row.update(overrides)
        return row

    def test_campaign_isolation_best_keep_score(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tsv = pathlib.Path(td) / "results.tsv"
            runner.ensure_tsv_header(tsv)
            runner.append_tsv(tsv, self._row(campaign_id="camp_a", stage="holdout", score="10.0"))
            runner.append_tsv(tsv, self._row(campaign_id="camp_b", stage="holdout", score="5.0"))
            runner.append_tsv(tsv, self._row(campaign_id="camp_b", stage="train", score="99.0"))
            runner.append_tsv(tsv, self._row(campaign_id="camp_b", stage="holdout", status="discard", suggestion="discard", score="20.0"))

            self.assertEqual(runner.read_best_keep_score(tsv, campaign_id="camp_a", stage="holdout"), 10.0)
            self.assertEqual(runner.read_best_keep_score(tsv, campaign_id="camp_b", stage="holdout"), 5.0)

    def test_absolute_hard_gates(self) -> None:
        ok_pair = {"ok": True, "reason": "ok"}

        hard_ok, reason = runner.evaluate_hard_gates(
            profit_pct=-0.1,
            drawdown_pct=5.0,
            sharpe=1.0,
            pair_gate=ok_pair,
            min_profit_pct=0.0,
            max_drawdown_pct=12.0,
            min_sharpe=0.0,
        )
        self.assertFalse(hard_ok)
        self.assertEqual(reason, "profit_below_min")

        hard_ok, reason = runner.evaluate_hard_gates(
            profit_pct=1.0,
            drawdown_pct=13.0,
            sharpe=1.0,
            pair_gate=ok_pair,
            min_profit_pct=0.0,
            max_drawdown_pct=12.0,
            min_sharpe=0.0,
        )
        self.assertFalse(hard_ok)
        self.assertEqual(reason, "drawdown_above_max")

        hard_ok, reason = runner.evaluate_hard_gates(
            profit_pct=1.0,
            drawdown_pct=5.0,
            sharpe=-0.1,
            pair_gate=ok_pair,
            min_profit_pct=0.0,
            max_drawdown_pct=12.0,
            min_sharpe=0.0,
        )
        self.assertFalse(hard_ok)
        self.assertEqual(reason, "sharpe_below_min")

    def test_pair_coverage_for_varied_pairlist_lengths(self) -> None:
        # One pair: threshold=max(10, ceil(100/(2*1))=50) -> pass with 60.
        one_pair = runner.evaluate_pair_coverage(
            total_trades=100,
            results_per_pair=[{"key": "BTC/USDT:USDT", "trades": 60}],
            pairlist=["BTC/USDT:USDT"],
            pair_min_trades_floor=10,
            pair_min_trades_mode="dynamic",
        )
        self.assertTrue(one_pair["ok"])
        self.assertEqual(one_pair["threshold"], 50)

        # Two pairs: threshold=max(10, ceil(100/(2*2))=25) -> fail because ETH has 20.
        two_pairs = runner.evaluate_pair_coverage(
            total_trades=100,
            results_per_pair=[
                {"key": "BTC/USDT:USDT", "trades": 80},
                {"key": "ETH/USDT:USDT", "trades": 20},
                {"key": "TOTAL", "trades": 100},
            ],
            pairlist=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            pair_min_trades_floor=10,
            pair_min_trades_mode="dynamic",
        )
        self.assertFalse(two_pairs["ok"])
        self.assertEqual(two_pairs["threshold"], 25)
        self.assertIn("ETH/USDT:USDT", two_pairs["missing_pairs"])

        # Four pairs: threshold=max(10, ceil(80/(2*4))=10) -> pass with 20 each.
        four_pairs = runner.evaluate_pair_coverage(
            total_trades=80,
            results_per_pair=[
                {"key": "A", "trades": 20},
                {"key": "B", "trades": 20},
                {"key": "C", "trades": 20},
                {"key": "D", "trades": 20},
                {"key": "TOTAL", "trades": 80},
            ],
            pairlist=["A", "B", "C", "D"],
            pair_min_trades_floor=10,
            pair_min_trades_mode="dynamic",
        )
        self.assertTrue(four_pairs["ok"])
        self.assertEqual(four_pairs["threshold"], 10)

    def test_parser_extracts_strategy_fields_from_json(self) -> None:
        payload = {
            "strategy": {
                "AutoresearchFreqAIStrategy": {
                    "profit_total": 0.12,
                    "max_drawdown_account": 0.08,
                    "sharpe": 1.2,
                    "total_trades": 24,
                    "pairlist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                    "results_per_pair": [
                        {"key": "BTC/USDT:USDT", "trades": 12},
                        {"key": "ETH/USDT:USDT", "trades": 12},
                    ],
                    "exit_reason_summary": [{"key": "roi", "trades": 8}],
                }
            },
            "strategy_comparison": [
                {
                    "key": "AutoresearchFreqAIStrategy",
                    "profit_total_pct": 12.0,
                    "max_drawdown_account": 0.08,
                    "sharpe": 1.2,
                    "trades": 24,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "bt.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            metrics = runner.parse_metrics_from_result_json(p, "AutoresearchFreqAIStrategy")

        self.assertEqual(metrics["profit_pct"], 12.0)
        self.assertEqual(metrics["max_drawdown_pct"], 8.0)
        self.assertEqual(metrics["sharpe"], 1.2)
        self.assertEqual(metrics["trades"], 24.0)
        self.assertEqual(metrics["pairlist"], ["BTC/USDT:USDT", "ETH/USDT:USDT"])
        self.assertEqual(len(metrics["results_per_pair"]), 2)
        self.assertEqual(len(metrics["exit_reason_summary"]), 1)

    def test_parser_reads_meta_zip_artifact(self) -> None:
        payload = {
            "strategy": {
                "AutoresearchFreqAIStrategy": {
                    "profit_total": 0.01,
                    "max_drawdown_account": 0.05,
                    "sharpe": 0.5,
                    "total_trades": 11,
                    "pairlist": ["BTC/USDT:USDT"],
                    "results_per_pair": [{"key": "BTC/USDT:USDT", "trades": 11}],
                    "exit_reason_summary": [],
                }
            },
            "strategy_comparison": [
                {
                    "key": "AutoresearchFreqAIStrategy",
                    "profit_total_pct": 1.0,
                    "max_drawdown_account": 0.05,
                    "sharpe": 0.5,
                    "trades": 11,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            meta = root / "backtest-result-2026-03-10_17-00-00.meta.json"
            zpath = root / "backtest-result-2026-03-10_17-00-00.zip"
            meta.write_text(json.dumps({"AutoresearchFreqAIStrategy": {"run_id": "x"}}), encoding="utf-8")
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("backtest-result-2026-03-10_17-00-00.json", json.dumps(payload))
                zf.writestr("backtest-result-2026-03-10_17-00-00_config.json", "{}")

            metrics = runner.parse_metrics_from_result_json(meta, "AutoresearchFreqAIStrategy")

        self.assertEqual(metrics["profit_pct"], 1.0)
        self.assertEqual(metrics["trades"], 11.0)

    def test_discover_latest_zip_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            freqtrade_dir = pathlib.Path(td)
            bt_dir = freqtrade_dir / "user_data" / "backtest_results"
            bt_dir.mkdir(parents=True)
            older = bt_dir / "backtest-result-2026-03-10_17-00-00.zip"
            newer = bt_dir / "backtest-result-2026-03-10_17-01-00.zip"
            older.write_text("x", encoding="utf-8")
            newer.write_text("x", encoding="utf-8")
            older.touch()
            newer.touch()

            found = runner.discover_backtest_artifact(
                log_text="",
                freqtrade_dir=freqtrade_dir,
                preferred=freqtrade_dir / "missing.json",
            )

        self.assertEqual(found.name, newer.name)


if __name__ == "__main__":
    unittest.main()
