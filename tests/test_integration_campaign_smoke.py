from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import freqai_autoresearch_loop as loop
import run_freqtrade_backtest as runner


class CampaignSmokeTest(unittest.TestCase):
    def test_train_holdout_rows_share_candidate_and_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tsv = pathlib.Path(td) / "results.tsv"
            runner.ensure_tsv_header(tsv)

            common = {
                "timestamp": "2026-03-10T00:00:00+00:00",
                "commit": "abc123",
                "campaign_id": "camp_xyz",
                "candidate_id": "cand_001",
                "timerange": "20250101-20250301",
                "config_fingerprint": "cfg",
                "pairlist_fingerprint": "pair",
                "profit_pct": "1.0",
                "max_drawdown_pct": "5.0",
                "sharpe": "0.5",
                "trades": "100",
                "score": "-1.5",
                "status": "keep",
                "suggestion": "keep",
                "decision_reason": "test",
                "description": "smoke",
                "log_path": "run.log",
                "json_path": "run.json",
            }

            row_train = dict(common)
            row_train["stage"] = "train"
            row_holdout = dict(common)
            row_holdout["stage"] = "holdout"

            runner.append_tsv(tsv, row_train)
            runner.append_tsv(tsv, row_holdout)

            train_row = loop.read_last_matching_row(tsv, "camp_xyz", "cand_001", "train")
            holdout_row = loop.read_last_matching_row(tsv, "camp_xyz", "cand_001", "holdout")

            self.assertIsNotNone(train_row)
            self.assertIsNotNone(holdout_row)
            self.assertEqual(train_row["campaign_id"], holdout_row["campaign_id"])
            self.assertEqual(train_row["candidate_id"], holdout_row["candidate_id"])
            self.assertEqual(train_row["stage"], "train")
            self.assertEqual(holdout_row["stage"], "holdout")

    def test_iteration_sequence_hyperopt_then_train_then_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            freqtrade_dir = root / "freqtrade"
            freqtrade_dir.mkdir(parents=True)

            strategy_file = root / "AutoresearchFreqAIStrategy.py"
            strategy_file.write_text(
                "\n".join(
                    [
                        "long_entry_ret = 0.011",
                        "short_entry_ret = -0.005",
                        "long_prob_min = 0.58",
                        "short_prob_max = 0.44",
                        "long_exit_prob_max = 0.4",
                        "short_exit_prob_min = 0.54",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "strategy_file": str(strategy_file),
                        "tunables": {
                            "long_prob_min": {
                                "min": 0.50,
                                "max": 0.75,
                                "step": 0.01,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            review_log = root / "review.jsonl"
            results_tsv = root / "results.tsv"
            space_state = root / "space_state.json"
            order: list[str] = []

            def fake_hyperopt(**kwargs):
                order.append("hyperopt")
                return 0

            def fake_backtest(**kwargs):
                stage = kwargs.get("stage")
                order.append(f"backtest:{stage}")
                return subprocess.CompletedProcess(args=["backtest"], returncode=0)

            def fake_last_row(_tsv, _campaign, _candidate, stage):
                if stage == "train":
                    return {"suggestion": "keep", "status": "keep", "decision_reason": "train_passed_hard_gates"}
                if stage == "holdout":
                    return {"suggestion": "keep", "status": "keep", "decision_reason": "first_holdout_keep"}
                return {}

            argv = [
                "freqai_autoresearch_loop.py",
                "--freqtrade-dir",
                str(freqtrade_dir),
                "--config",
                "user_data/config.json",
                "--train-timerange",
                "20250101-20250301",
                "--holdout-timerange",
                "20250301-20250501",
                "--iterations",
                "1",
                "--seed",
                "42",
                "--profile",
                str(profile_path),
                "--review-log",
                str(review_log),
                "--results-tsv",
                str(results_tsv),
                "--space-state",
                str(space_state),
                "--repo-dir",
                str(root),
                "--no-space-adaptation",
            ]

            with mock.patch.object(loop, "run_hyperopt", side_effect=fake_hyperopt):
                with mock.patch.object(loop, "run_backtest_runner", side_effect=fake_backtest):
                    with mock.patch.object(loop, "read_last_matching_row", side_effect=fake_last_row):
                        with mock.patch.object(sys, "argv", argv):
                            rc = loop.main()

            self.assertEqual(rc, 0)
            self.assertEqual(order[:3], ["hyperopt", "backtest:train", "backtest:holdout"])

    def test_iteration_sequence_llm_then_hyperopt_then_train_then_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            freqtrade_dir = root / "freqtrade"
            freqtrade_dir.mkdir(parents=True)

            strategy_file = root / "AutoresearchFreqAIStrategy.py"
            strategy_file.write_text(
                "\n".join(
                    [
                        "long_entry_ret = 0.011",
                        "short_entry_ret = -0.005",
                        "long_prob_min = 0.58",
                        "short_prob_max = 0.44",
                        "long_exit_prob_max = 0.4",
                        "short_exit_prob_min = 0.54",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            model_file = root / "freqtrade" / "freqaimodels" / "AutoresearchLSTMRegressor.py"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.write_text("x = 1\n", encoding="utf-8")

            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "strategy_file": str(strategy_file),
                        "tunables": {
                            "long_prob_min": {
                                "min": 0.50,
                                "max": 0.75,
                                "step": 0.01,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cred_file = root / "openai.json"
            cred_file.write_text(json.dumps({"api_key": "sk-test"}), encoding="utf-8")

            review_log = root / "review.jsonl"
            results_tsv = root / "results.tsv"
            space_state = root / "space_state.json"
            order: list[str] = []

            def fake_llm(**kwargs):
                order.append("llm")
                return {
                    "ran": True,
                    "model": "gpt-5-mini",
                    "request_id": "resp_x",
                    "prompt_hash": "abc",
                    "patch_applied": True,
                    "repair_attempts_used": 0,
                    "error": None,
                    "patch_text": "diff --git ...",
                    "usage": None,
                    "upstream_ref": None,
                    "upstream_commit": None,
                }

            def fake_hyperopt(**kwargs):
                order.append("hyperopt")
                return 0

            def fake_backtest(**kwargs):
                stage = kwargs.get("stage")
                order.append(f"backtest:{stage}")
                return subprocess.CompletedProcess(args=["backtest"], returncode=0)

            def fake_last_row(_tsv, _campaign, _candidate, stage):
                if stage == "train":
                    return {"suggestion": "keep", "status": "keep", "decision_reason": "train_passed_hard_gates"}
                if stage == "holdout":
                    return {"suggestion": "keep", "status": "keep", "decision_reason": "first_holdout_keep"}
                return {}

            argv = [
                "freqai_autoresearch_loop.py",
                "--freqtrade-dir",
                str(freqtrade_dir),
                "--config",
                "user_data/config.json",
                "--train-timerange",
                "20250101-20250301",
                "--holdout-timerange",
                "20250301-20250501",
                "--iterations",
                "1",
                "--seed",
                "42",
                "--profile",
                str(profile_path),
                "--review-log",
                str(review_log),
                "--results-tsv",
                str(results_tsv),
                "--space-state",
                str(space_state),
                "--repo-dir",
                str(root),
                "--llm-enable",
                "--llm-credentials-file",
                str(cred_file),
            ]

            with mock.patch.object(loop, "run_llm_patch_stage", side_effect=fake_llm):
                with mock.patch.object(loop, "run_hyperopt", side_effect=fake_hyperopt):
                    with mock.patch.object(loop, "run_backtest_runner", side_effect=fake_backtest):
                        with mock.patch.object(loop, "read_last_matching_row", side_effect=fake_last_row):
                            with mock.patch.object(sys, "argv", argv):
                                rc = loop.main()

            self.assertEqual(rc, 0)
            self.assertEqual(order[:4], ["llm", "hyperopt", "backtest:train", "backtest:holdout"])


if __name__ == "__main__":
    unittest.main()
