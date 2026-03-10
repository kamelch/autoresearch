from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import llm_patch_engine as eng


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http_{self.status_code}")

    def json(self) -> dict:
        return self._payload


class LLMPatchEngineTests(unittest.TestCase):
    def test_load_openai_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cred = pathlib.Path(td) / "openai.json"
            cred.write_text(
                json.dumps(
                    {
                        "api_key": "sk-test",
                        "base_url": "https://example.invalid/v1",
                        "organization": "org_x",
                        "project": "proj_x",
                    }
                ),
                encoding="utf-8",
            )
            loaded = eng.load_openai_credentials(cred)
            self.assertEqual(loaded["api_key"], "sk-test")
            self.assertEqual(loaded["base_url"], "https://example.invalid/v1")
            self.assertEqual(loaded["organization"], "org_x")
            self.assertEqual(loaded["project"], "proj_x")

    def test_load_openai_credentials_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cred = pathlib.Path(td) / "openai.json"
            cred.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                eng.load_openai_credentials(cred)

    def test_extract_and_validate_unified_diff(self) -> None:
        patch = """```diff
diff --git a/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py b/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py
--- a/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py
+++ b/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py
@@ -1,1 +1,1 @@
-x = 1
+x = 2
```"""
        extracted = eng.extract_unified_diff(patch)
        self.assertTrue(extracted.startswith("diff --git"))

        ok, err, touched = eng.validate_patch_targets(
            extracted,
            "freqtrade/freqaimodels/AutoresearchLSTMRegressor.py",
        )
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertEqual(touched, ["freqtrade/freqaimodels/AutoresearchLSTMRegressor.py"])

        bad_patch = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x
+y
"""
        ok, err, _ = eng.validate_patch_targets(
            bad_patch,
            "freqtrade/freqaimodels/AutoresearchLSTMRegressor.py",
        )
        self.assertFalse(ok)
        assert err is not None
        self.assertIn("disallowed", err)

    def test_build_prompt_payload_contains_context(self) -> None:
        system_prompt, user_prompt, prompt_hash = eng.build_prompt_payload(
            target_relative_path="freqtrade/freqaimodels/AutoresearchLSTMRegressor.py",
            target_source="class X:\n    pass\n",
            recent_results=[{"stage": "train", "suggestion": "discard"}],
            recent_review=[{"final_suggestion": "discard"}],
            upstream_bundle={"ref": "main", "commit": "abc", "snippets": {"program.md": "..."}, "error": None},
            repair_hint="py_compile_failed",
            previous_patch="diff --git ...",
        )
        self.assertIn("Output ONLY a unified diff", system_prompt)
        self.assertIn("CURRENT FILE:", user_prompt)
        self.assertIn("RECENT RESULTS", user_prompt)
        self.assertIn("UPSTREAM CONTEXT", user_prompt)
        self.assertIn("PREVIOUS ATTEMPT ERROR", user_prompt)
        self.assertTrue(len(prompt_hash) == 64)

    def test_generate_patch_with_openai_parses_response(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = pathlib.Path(td)
            creds_path = td_path / "openai.json"
            creds_path.write_text(json.dumps({"api_key": "sk-test"}), encoding="utf-8")

            tsv = td_path / "results.tsv"
            tsv.write_text(
                "\t".join(["campaign_id", "candidate_id", "stage", "suggestion"]) + "\n"
                + "\t".join(["camp_x", "cand_1", "train", "discard"]) + "\n",
                encoding="utf-8",
            )
            review = td_path / "review.jsonl"
            review.write_text(json.dumps({"campaign_id": "camp_x", "candidate_id": "cand_1"}) + "\n", encoding="utf-8")

            fake_payload = {
                "id": "resp_123",
                "output_text": (
                    "diff --git a/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py "
                    "b/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py\n"
                    "--- a/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py\n"
                    "+++ b/freqtrade/freqaimodels/AutoresearchLSTMRegressor.py\n"
                    "@@ -1 +1 @@\n"
                    "-x = 1\n"
                    "+x = 2\n"
                ),
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

            with mock.patch.object(eng.requests, "post", return_value=_FakeResponse(200, payload=fake_payload)):
                result = eng.generate_patch_with_openai(
                    credentials_file=creds_path,
                    model="gpt-5-mini",
                    target_relative_path="freqtrade/freqaimodels/AutoresearchLSTMRegressor.py",
                    target_source="x = 1\n",
                    results_tsv=tsv,
                    review_log=review,
                    campaign_id="camp_x",
                    context_rows=10,
                    timeout_sec=10,
                )

            self.assertEqual(result["request_id"], "resp_123")
            self.assertIn("diff --git", result["patch_text"])
            self.assertIsInstance(result.get("prompt_hash"), str)

    def test_fetch_upstream_context_bundle_cached(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_dir = pathlib.Path(td)

            def fake_get(url: str, *args, **kwargs):
                if "api.github.com" in url:
                    return _FakeResponse(200, payload={"sha": "abc123"})
                return _FakeResponse(200, payload={}, text=f"content:{url}")

            with mock.patch.object(eng.requests, "get", side_effect=fake_get):
                bundle = eng.fetch_upstream_context_bundle(cache_dir, ref="main", timeout_sec=5)

            self.assertEqual(bundle["ref"], "main")
            self.assertEqual(bundle["commit"], "abc123")
            self.assertIn("README.md", bundle["snippets"])
            self.assertTrue((cache_dir / "upstream_main.json").exists())


if __name__ == "__main__":
    unittest.main()
