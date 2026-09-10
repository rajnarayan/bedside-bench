#!/usr/bin/env python3

import unittest

from evaluation.run_benchmark import (
    dataset_sha256,
    parse_decisions,
    score_case,
    summarize,
)


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


if __name__ == "__main__":
    unittest.main()
