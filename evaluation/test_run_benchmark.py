#!/usr/bin/env python3

import os
import unittest
from unittest.mock import patch

from evaluation.run_benchmark import (
    NEOMD_TARGETS,
    dataset_sha256,
    is_neomd_model,
    parse_decisions,
    score_case,
    summarize,
)
from evaluation.run_benchmark import _neomd_email, _neomd_password, _neomd_target


class ScoreCaseTest(unittest.TestCase):
    def test_points_rubric_normalizes_and_applies_penalty(self):
        case = {
            "grading_method": "points_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 2},
                {"item_id": "b", "points": 1},
                {"item_id": "c", "points": -1},
            ],
        }
        score, components = score_case(case, {"a": True, "b": False, "c": True})
        self.assertAlmostEqual(score, 1 / 3)
        self.assertEqual(components["matched_signed_points"], 1)
        self.assertEqual(components["available_positive_points"], 3)

    def test_points_rubric_clamps_below_zero(self):
        case = {
            "grading_method": "points_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 1},
                {"item_id": "b", "points": -2},
            ],
        }
        score, _ = score_case(case, {"a": False, "b": True})
        self.assertEqual(score, 0)

    def test_f1_weighted_rubric(self):
        case = {
            "grading_method": "f1_weighted_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 3},
                {"item_id": "b", "points": 1},
                {"item_id": "c", "points": -2},
                {"item_id": "d", "points": 0},
            ],
        }
        score, components = score_case(
            case, {"a": True, "b": False, "c": True, "d": True}
        )
        # +3/+1/−2/0 → classes 9/7/2/5 → weights 72/3/24/1.
        # Recall 72/75; precision 72/97.
        recall = 72 / 75
        precision = 72 / 97
        expected = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(score, round(expected, 4))
        self.assertEqual(components["severe_rate"], 0.0)
        self.assertAlmostEqual(components["recall_weighted"], round(recall, 4))
        self.assertAlmostEqual(components["precision_weighted"], round(precision, 4))

    def test_f1_weighted_rubric_severe_omission_is_reported_not_gated(self):
        case = {
            "grading_method": "f1_weighted_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 3},
                {"item_id": "b", "points": 2},
            ],
        }
        score, components = score_case(case, {"a": False, "b": True})
        # +3/+2 → classes 9/8 → weights 72/24. Missing "a" is a severe
        # omission: it is reported, but the weighted F1 stands on its own.
        recall = 24 / 96
        precision = 1.0
        expected = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(score, round(expected, 4))
        self.assertEqual(components["severe_rate"], 1.0)

    def test_f1_weighted_rubric_severe_commission_is_reported_not_gated(self):
        case = {
            "grading_method": "f1_weighted_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 2},
                {"item_id": "b", "points": -3},
            ],
        }
        score, components = score_case(case, {"a": True, "b": True})
        # +2/−3 → classes 8/1 → weights 24/72. Asserting "b" is a severe
        # commission: it lands in the precision denominator, not a gate.
        recall = 1.0
        precision = 24 / 96
        expected = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(score, round(expected, 4))
        self.assertEqual(components["severe_rate"], 1.0)

    def test_judge_decisions_must_cover_exact_item_ids(self):
        case = {
            "rubric_items": [
                {"item_id": "a"},
                {"item_id": "b"},
            ]
        }
        with self.assertRaises(ValueError):
            parse_decisions(
                case,
                '{"decisions":[{"item_id":"a","matched":true}]}',
            )

    def test_dataset_digest_uses_canonical_case_content(self):
        first = [{"case_id": "a", "prompt": "test", "rubric_items": []}]
        reordered = [{"rubric_items": [], "prompt": "test", "case_id": "a"}]
        self.assertEqual(dataset_sha256(first), dataset_sha256(reordered))

    def test_summary_includes_reproducibility_metadata(self):
        rows = [
            {
                "case_id": "a",
                "benchmark": "example",
                "score": 0.5,
                "answer_model": "answer",
                "judge_model": "judge",
                "answer_temperature": 0,
                "judge_temperature": None,
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:01:00+00:00",
            }
        ]
        summary = summarize(rows, 10, 42, {"a"}, "digest")
        self.assertEqual(summary["dataset_sha256"], "digest")
        self.assertEqual(summary["answer_temperature"], 0)
        self.assertEqual(summary["judge_passes_per_answer"], 1)
        self.assertEqual(summary["errors"], 0)


class NeomdProviderTest(unittest.TestCase):
    """NeoMD is referenced by name string with no plugin architecture in this
    repo, so target/credential resolution is hand-rolled -- these pin that
    resolution logic without making a real network call."""

    def test_only_the_two_named_targets_are_neomd_models(self):
        self.assertTrue(is_neomd_model("neomd-local"))
        self.assertTrue(is_neomd_model("neomd-prod"))
        self.assertFalse(is_neomd_model("gpt-5.6-sol"))
        self.assertFalse(is_neomd_model("claude-opus-5"))

    def test_unknown_neomd_target_raises_with_the_known_list(self):
        with self.assertRaises(SystemExit) as ctx:
            _neomd_target("neomd-staging")
        self.assertIn("neomd-local", str(ctx.exception))
        self.assertIn("neomd-prod", str(ctx.exception))

    def test_local_target_uses_inline_test_email(self):
        target = NEOMD_TARGETS["neomd-local"]
        self.assertEqual(_neomd_email(target), "test@neomd.ai")

    def test_prod_target_requires_neomd_email_env_var(self):
        target = NEOMD_TARGETS["neomd-prod"]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEOMD_EMAIL", None)
            with self.assertRaises(SystemExit):
                _neomd_email(target)
        with patch.dict(os.environ, {"NEOMD_EMAIL": "raj@kanza.ai"}):
            self.assertEqual(_neomd_email(target), "raj@kanza.ai")

    def test_password_env_var_missing_raises(self):
        target = NEOMD_TARGETS["neomd-local"]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEOMD_CHAT_PASSWORD", None)
            with self.assertRaises(SystemExit):
                _neomd_password(target)


if __name__ == "__main__":
    unittest.main()
