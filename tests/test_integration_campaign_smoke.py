from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
