from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import automate_freqai_pipeline as pipeline
import freqai_autoresearch_loop as loop
import run_freqtrade_backtest as runner


def _flag_value(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


class LSTMDefaultScriptTests(unittest.TestCase):
    def test_parser_defaults_use_lstm_model(self) -> None:
        pipeline_args = pipeline.build_parser().parse_args(
            [
                "--freqtrade-dir",
                "/tmp/freqtrade",
                "--train-timerange",
                "20250101-20250301",
                "--holdout-timerange",
                "20250301-20250501",
            ]
        )
        self.assertEqual(pipeline_args.freqaimodel, "AutoresearchLSTMRegressor")

        loop_args = loop.build_parser().parse_args(
            [
                "--freqtrade-dir",
                "/tmp/freqtrade",
                "--config",
                "user_data/config.json",
                "--train-timerange",
                "20250101-20250301",
                "--holdout-timerange",
                "20250301-20250501",
            ]
        )
        self.assertEqual(loop_args.freqaimodel, "AutoresearchLSTMRegressor")

        runner_args = runner.build_parser().parse_args(
            [
                "--freqtrade-dir",
                "/tmp/freqtrade",
                "--config",
                "user_data/config.json",
                "--strategy",
                "AutoresearchFreqAIStrategy",
                "--timerange",
                "20250101-20250301",
            ]
        )
        self.assertEqual(runner_args.freqaimodel, "AutoresearchLSTMRegressor")

    def test_loop_runner_passes_freqaimodel_path_and_overrides(self) -> None:
        captured: dict[str, list[str]] = {}

        def fake_run(cmd: list[str], text: bool, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.object(loop.subprocess, "run", side_effect=fake_run):
            proc = loop.run_backtest_runner(
                repo_dir=pathlib.Path("/tmp/repo"),
                freqtrade_dir=pathlib.Path("/tmp/freqtrade"),
                config="user_data/config.json",
                freqtrade_bin="/tmp/freqtrade/.venv/bin/freqtrade",
                strategy="AutoresearchFreqAIStrategy",
                strategy_path="/tmp/repo/freqtrade/strategies",
                freqaimodel="AmazonChronos2Regressor",
                freqaimodel_path="/tmp/repo/freqtrade/freqaimodels",
                timerange="20250101-20250301",
                description="test",
                results_tsv=pathlib.Path("/tmp/repo/freqtrade/results/results_x.tsv"),
                dd_penalty=0.5,
                min_improvement=0.0,
                campaign_id="camp_x",
                candidate_id="cand_x",
                stage="holdout",
                min_profit_pct=0.0,
                max_drawdown_pct=12.0,
                min_sharpe=0.0,
                pair_min_trades_floor=10,
                pair_min_trades_mode="dynamic",
            )

        self.assertEqual(proc.returncode, 0)
        cmd = captured["cmd"]
        self.assertIn("--freqaimodel", cmd)
        self.assertEqual(_flag_value(cmd, "--freqaimodel"), "AmazonChronos2Regressor")
        self.assertIn("--freqaimodel-path", cmd)
        self.assertEqual(_flag_value(cmd, "--freqaimodel-path"), "/tmp/repo/freqtrade/freqaimodels")

    def test_pipeline_passes_freqaimodel_path_and_overrides(self) -> None:
        captured_cmds: list[list[str]] = []

        def fake_run_cmd(cmd: list[str], cwd: pathlib.Path) -> int:
            captured_cmds.append(cmd)
            return 0

        with tempfile.TemporaryDirectory() as td:
            freqtrade_dir = pathlib.Path(td) / "freqtrade"
            freqtrade_dir.mkdir(parents=True)

            argv = [
                "automate_freqai_pipeline.py",
                "--freqtrade-dir",
                str(freqtrade_dir),
                "--train-timerange",
                "20250101-20250301",
                "--holdout-timerange",
                "20250301-20250501",
                "--iterations",
                "1",
                "--skip-download",
                "--freqaimodel",
                "AmazonChronos2Regressor",
                "--freqaimodel-path",
                "/tmp/repo/freqtrade/freqaimodels",
                "--llm-enable",
                "--llm-model",
                "gpt-5-mini",
                "--llm-credentials-file",
                "/tmp/openai.json",
                "--llm-repair-attempts",
                "2",
                "--llm-context-rows",
                "10",
                "--llm-timeout-sec",
                "60",
                "--llm-sync-upstream",
                "--llm-upstream-ref",
                "main",
            ]

            with mock.patch.object(pipeline, "run_cmd", side_effect=fake_run_cmd):
                with mock.patch.object(pipeline, "summarize_results", return_value=None):
                    with mock.patch.object(sys, "argv", argv):
                        rc = pipeline.main()

        self.assertEqual(rc, 0)
        self.assertGreaterEqual(len(captured_cmds), 3)

        hyperopt_cmd = next(cmd for cmd in captured_cmds if "hyperopt" in cmd)

        baseline_cmd = next(
            cmd for cmd in captured_cmds if any("run_freqtrade_backtest.py" in token for token in cmd)
        )
        loop_cmd = next(
            cmd for cmd in captured_cmds if any("freqai_autoresearch_loop.py" in token for token in cmd)
        )

        self.assertIn("--freqaimodel", hyperopt_cmd)
        self.assertEqual(_flag_value(hyperopt_cmd, "--freqaimodel"), "AmazonChronos2Regressor")
        self.assertIn("--freqaimodel-path", hyperopt_cmd)
        self.assertEqual(_flag_value(hyperopt_cmd, "--freqaimodel-path"), "/tmp/repo/freqtrade/freqaimodels")

        self.assertEqual(_flag_value(baseline_cmd, "--freqaimodel"), "AmazonChronos2Regressor")
        self.assertEqual(_flag_value(baseline_cmd, "--freqaimodel-path"), "/tmp/repo/freqtrade/freqaimodels")
        self.assertEqual(_flag_value(loop_cmd, "--freqaimodel"), "AmazonChronos2Regressor")
        self.assertEqual(_flag_value(loop_cmd, "--freqaimodel-path"), "/tmp/repo/freqtrade/freqaimodels")
        self.assertEqual(_flag_value(loop_cmd, "--hyperopt-cadence"), "5")
        self.assertEqual(_flag_value(loop_cmd, "--hyperopt-loss"), "ProfitDrawDownHyperOptLoss")
        self.assertIn("--llm-enable", loop_cmd)
        self.assertEqual(_flag_value(loop_cmd, "--llm-model"), "gpt-5-mini")
        self.assertEqual(_flag_value(loop_cmd, "--llm-credentials-file"), "/tmp/openai.json")
        self.assertEqual(_flag_value(loop_cmd, "--llm-upstream-ref"), "main")
        self.assertIn("--llm-sync-upstream", loop_cmd)


if __name__ == "__main__":
    unittest.main()
