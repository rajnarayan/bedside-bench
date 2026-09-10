"""Tests for globally unique ``md5(prompt)_{index}`` item IDs."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from item_ids import item_id_for, question_md5

ROOT = Path(__file__).resolve().parents[1]


class ItemIdTests(unittest.TestCase):
    def test_formula(self):
        prompt = "example question"
        self.assertEqual(item_id_for(prompt, 0), f"{question_md5(prompt)}_0")
        self.assertNotEqual(item_id_for(prompt, 0), item_id_for(prompt, 1))

    def test_panel_item_ids_are_globally_unique_and_match_formula(self):
        ids: list[str] = []
        for path in (ROOT / "data").glob("*/cases/*.json"):
            case = json.loads(path.read_text())
            prompt = case["prompt"]
            for index, item in enumerate(case["rubric_items"]):
                expected = item_id_for(prompt, index)
                self.assertEqual(item["item_id"], expected, path.name)
                ids.append(item["item_id"])
        counts = Counter(ids)
        self.assertEqual(len(ids), len(counts), dict(counts.most_common(3)))


if __name__ == "__main__":
    unittest.main()
