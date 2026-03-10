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
from autoresearch_utils import exclusive_lock


class LoopStateTests(unittest.TestCase):
    def _tunables(self) -> dict[str, dict[str, float]]:
        return {
            "x": {
                "min": 0.1,
                "max": 0.5,
                "step": 0.1,
                "hard_min": 0.0,
                "hard_max": 1.0,
            }
        }

    def _profile_raw(self) -> dict[str, object]:
        return {
            "strategy_file": "freqtrade/strategies/AutoresearchFreqAIStrategy.py",
            "tunables": {
                "x": {
                    "min": 0.1,
                    "max": 0.5,
                    "step": 0.1,
                }
            },
        }

    def test_state_reset_on_metadata_mismatch(self) -> None:
        tunables = self._tunables()
        profile_raw = self._profile_raw()

        with tempfile.TemporaryDirectory() as td:
            state_path = pathlib.Path(td) / "space_state.json"
            old_state = {
                "version": 2,
                "campaign_id": "old_campaign",
                "profile_hash": "old_hash",
                "profile_path": "/tmp/old_profile.json",
                "params": {
                    "x": {
                        "hard_min": 0.0,
                        "hard_max": 1.0,
                        "active_min": 0.4,
                        "active_max": 0.8,
                        "step": 0.1,
                        "weight": 2.0,
                        "keep_count": 3,
                        "discard_count": 1,
                        "crash_count": 0,
                    }
                },
            }
            state_path.write_text(json.dumps(old_state), encoding="utf-8")

            reset_state, reason = loop.load_or_init_space_state(
                path=state_path,
                tunables=tunables,
                profile_raw=profile_raw,
                campaign_id="new_campaign",
                profile_path=pathlib.Path("/tmp/new_profile.json"),
                profile_hash="new_hash",
                reuse_space_state=False,
                reset_space_state=False,
            )
            self.assertEqual(reason, "metadata_mismatch_reset")
            self.assertEqual(reset_state["params"]["x"]["active_min"], 0.1)
            self.assertEqual(reset_state["params"]["x"]["active_max"], 0.5)

            reused_state, reason = loop.load_or_init_space_state(
                path=state_path,
                tunables=tunables,
                profile_raw=profile_raw,
                campaign_id="new_campaign",
                profile_path=pathlib.Path("/tmp/new_profile.json"),
                profile_hash="new_hash",
                reuse_space_state=True,
                reset_space_state=False,
            )
            self.assertEqual(reason, "metadata_mismatch_reused")
            self.assertEqual(reused_state["params"]["x"]["active_min"], 0.4)
            self.assertEqual(reused_state["params"]["x"]["active_max"], 0.8)

    def test_choose_final_suggestion_train_pass_holdout_fail(self) -> None:
        final_suggestion = loop.choose_final_suggestion(
            train_suggestion="keep",
            holdout_suggestion="discard",
            train_returncode=0,
            holdout_returncode=0,
        )
        self.assertEqual(final_suggestion, "discard")

    def test_lock_contention_raises_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lock_path = pathlib.Path(td) / "lockfile.lock"
            with exclusive_lock(lock_path):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(lock_path):
                        pass

    def test_restore_strategy_restores_committed_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            strategy_file = pathlib.Path(td) / "strategy.py"
            strategy_file.write_text("x = 1\n", encoding="utf-8")
            strategy_file.write_text("x = 2\n", encoding="utf-8")
            loop.restore_strategy(strategy_file, "x = 1\n")
            self.assertEqual(strategy_file.read_text(encoding="utf-8"), "x = 1\n")

    def test_should_run_hyperopt_cadence(self) -> None:
        cadence_hits = [i for i in range(1, 13) if loop.should_run_hyperopt(i, 5)]
        self.assertEqual(cadence_hits, [1, 6, 11])

    def test_run_hyperopt_command_wiring(self) -> None:
        captured: dict[str, object] = {}

        def fake_stream(cmd: list[str], cwd: pathlib.Path) -> int:
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            return 0

        with mock.patch.object(loop, "resolve_freqtrade_cmd", return_value=["/tmp/freqtrade"]):
            with mock.patch.object(loop, "run_stream_cmd", side_effect=fake_stream):
                rc = loop.run_hyperopt(
                    freqtrade_dir=pathlib.Path("/tmp/freqtrade_dir"),
                    freqtrade_bin="/tmp/freqtrade",
                    config="user_data/config.json",
                    strategy="AutoresearchFreqAIStrategy",
                    strategy_path="/tmp/strategies",
                    freqaimodel="AutoresearchLSTMRegressor",
                    freqaimodel_path="/tmp/freqaimodels",
                    timerange="20250101-20250301",
                    epochs=100,
                    spaces="all",
                    loss="ProfitDrawDownHyperOptLoss",
                    jobs=-1,
                    min_trades=1,
                    random_state=42,
                    ignore_missing_spaces=True,
                )

        self.assertEqual(rc, 0)
        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        self.assertIn("hyperopt", cmd)
        self.assertIn("--spaces", cmd)
        self.assertIn("all", cmd)
        self.assertIn("--hyperopt-loss", cmd)
        self.assertIn("ProfitDrawDownHyperOptLoss", cmd)
        self.assertIn("--ignore-missing-spaces", cmd)

    def test_discard_restores_strategy_and_param_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            strategy_file = root / "AutoresearchFreqAIStrategy.py"
            original_strategy_text = "\n".join(
                [
                    "long_entry_ret = 0.011",
                    "short_entry_ret = -0.005",
                    "long_prob_min = 0.58",
                    "short_prob_max = 0.44",
                    "long_exit_prob_max = 0.4",
                    "short_exit_prob_min = 0.54",
                    "",
                ]
            )
            strategy_file.write_text(original_strategy_text, encoding="utf-8")
            params_file = strategy_file.with_suffix(".json")
            params_file.write_text('{"params":{"buy":{"x":1}}}\n', encoding="utf-8")

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
            freqtrade_dir = root / "freqtrade"
            freqtrade_dir.mkdir(parents=True)

            def fake_backtest_runner(**kwargs):
                return subprocess.CompletedProcess(args=["backtest"], returncode=0)

            def fake_last_row(_tsv, _campaign, _candidate, stage):
                if stage == "train":
                    return {"suggestion": "discard", "status": "discard", "decision_reason": "score_below_best_keep"}
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
            ]

            with mock.patch.object(loop, "run_hyperopt", return_value=0):
                with mock.patch.object(loop, "run_backtest_runner", side_effect=fake_backtest_runner):
                    with mock.patch.object(loop, "read_last_matching_row", side_effect=fake_last_row):
                        with mock.patch.object(sys, "argv", argv):
                            rc = loop.main()

            self.assertEqual(rc, 0)
            self.assertEqual(strategy_file.read_text(encoding="utf-8"), original_strategy_text)
            self.assertEqual(params_file.read_text(encoding="utf-8"), '{"params":{"buy":{"x":1}}}\n')

    def test_run_llm_patch_stage_repair_then_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_file = root / "freqtrade" / "freqaimodels" / "AutoresearchLSTMRegressor.py"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.write_text("x = 1\n", encoding="utf-8")

            patch_text = (
                "diff --git a/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py b/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py\n"
                "--- a/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py\n"
                "+++ b/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py\n"
                "@@ -1 +1 @@\n"
                "-x = 1\n"
                "+x = 2\n"
            )

            call_count = {"n": 0}

            def fake_generate(**kwargs):
                call_count["n"] += 1
                return {
                    "request_id": f"resp_{call_count['n']}",
                    "prompt_hash": "abc",
                    "patch_text": patch_text,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }

            compile_results = [(False, "py_compile_failed"), (True, None)]

            with mock.patch.object(loop.llm_patch_engine, "generate_patch_with_openai", side_effect=fake_generate):
                with mock.patch.object(loop.llm_patch_engine, "apply_unified_diff", return_value=(True, None)):
                    with mock.patch.object(loop.llm_patch_engine, "run_py_compile", side_effect=compile_results):
                        info = loop.run_llm_patch_stage(
                            repo_dir=root,
                            model_file=model_file,
                            campaign_id="camp_x",
                            results_tsv=root / "results.tsv",
                            review_log=root / "review.jsonl",
                            llm_model="gpt-5-mini",
                            credentials_file=root / "openai.json",
                            context_rows=10,
                            timeout_sec=10,
                            repair_attempts=2,
                            upstream_bundle=None,
                        )

            self.assertTrue(info["patch_applied"])
            self.assertEqual(info["repair_attempts_used"], 1)
            self.assertEqual(call_count["n"], 2)

    def test_run_llm_patch_stage_rejects_disallowed_targets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_file = root / "freqtrade" / "freqaimodels" / "AutoresearchLSTMRegressor.py"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            original = "x = 1\n"
            model_file.write_text(original, encoding="utf-8")

            bad_patch = (
                "diff --git a/other.py b/other.py\n"
                "--- a/other.py\n"
                "+++ b/other.py\n"
                "@@ -1 +1 @@\n"
                "-x\n"
                "+y\n"
            )

            with mock.patch.object(
                loop.llm_patch_engine,
                "generate_patch_with_openai",
                return_value={"request_id": "resp_1", "prompt_hash": "abc", "patch_text": bad_patch, "usage": None},
            ):
                info = loop.run_llm_patch_stage(
                    repo_dir=root,
                    model_file=model_file,
                    campaign_id="camp_x",
                    results_tsv=root / "results.tsv",
                    review_log=root / "review.jsonl",
                    llm_model="gpt-5-mini",
                    credentials_file=root / "openai.json",
                    context_rows=10,
                    timeout_sec=10,
                    repair_attempts=0,
                    upstream_bundle=None,
                )

            self.assertFalse(info["patch_applied"])
            self.assertIn("disallowed", str(info["error"]))
            self.assertEqual(model_file.read_text(encoding="utf-8"), original)

    def test_llm_discard_restores_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            strategy_file = root / "AutoresearchFreqAIStrategy.py"
            original_strategy_text = "\n".join(
                [
                    "long_entry_ret = 0.011",
                    "short_entry_ret = -0.005",
                    "long_prob_min = 0.58",
                    "short_prob_max = 0.44",
                    "long_exit_prob_max = 0.4",
                    "short_exit_prob_min = 0.54",
                    "",
                ]
            )
            strategy_file.write_text(original_strategy_text, encoding="utf-8")
            params_file = strategy_file.with_suffix(".json")
            params_file.write_text('{"params":{"buy":{"x":1}}}\n', encoding="utf-8")

            model_file = root / "freqtrade" / "freqaimodels" / "AutoresearchLSTMRegressor.py"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            original_model_text = "x = 1\n"
            model_file.write_text(original_model_text, encoding="utf-8")

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
            freqtrade_dir = root / "freqtrade"
            freqtrade_dir.mkdir(parents=True, exist_ok=True)

            def fake_backtest_runner(**kwargs):
                return subprocess.CompletedProcess(args=["backtest"], returncode=0)

            def fake_last_row(_tsv, _campaign, _candidate, stage):
                if stage == "train":
                    return {"suggestion": "discard", "status": "discard", "decision_reason": "score_below_best_keep"}
                return {}

            def fake_llm_stage(**kwargs):
                model_file.write_text("x = 2\n", encoding="utf-8")
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

            cred_file = root / "openai.json"
            cred_file.write_text(json.dumps({"api_key": "sk-test"}), encoding="utf-8")

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

            with mock.patch.object(loop, "run_hyperopt", return_value=0):
                with mock.patch.object(loop, "run_backtest_runner", side_effect=fake_backtest_runner):
                    with mock.patch.object(loop, "read_last_matching_row", side_effect=fake_last_row):
                        with mock.patch.object(loop, "run_llm_patch_stage", side_effect=fake_llm_stage):
                            with mock.patch.object(sys, "argv", argv):
                                rc = loop.main()

            self.assertEqual(rc, 0)
            self.assertEqual(model_file.read_text(encoding="utf-8"), original_model_text)

    def test_upstream_sync_called_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
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
            freqtrade_dir = root / "freqtrade"
            freqtrade_dir.mkdir(parents=True, exist_ok=True)

            common_argv = [
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
                "0",
                "--profile",
                str(profile_path),
                "--repo-dir",
                str(root),
                "--llm-enable",
                "--llm-credentials-file",
                str(cred_file),
            ]

            with mock.patch.object(
                loop.llm_patch_engine, "fetch_upstream_context_bundle", return_value={"ref": "main"}
            ) as fetch_mock:
                with mock.patch.object(sys, "argv", common_argv):
                    rc = loop.main()
            self.assertEqual(rc, 0)
            fetch_mock.assert_not_called()

            with mock.patch.object(loop.llm_patch_engine, "fetch_upstream_context_bundle", return_value={"ref": "main"}) as fetch_mock:
                with mock.patch.object(sys, "argv", common_argv + ["--llm-sync-upstream"]):
                    rc = loop.main()
            self.assertEqual(rc, 0)
            fetch_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
