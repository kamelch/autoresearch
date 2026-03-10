from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
