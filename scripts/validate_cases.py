#!/usr/bin/env python3
"""Validate every case JSON against its grading-method schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from load_panel import load_panel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from item_ids import item_id_for  # noqa: E402

SCHEMA_BY_METHOD = {
    "points_rubric": "case.points_rubric.schema.json",
    "f1_weighted_rubric": "case.f1_weighted_rubric.schema.json",
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    schema_dir = root / "schemas"
    registry = json.loads((root / "registry.json").read_text())
    errors: list[str] = []
    checked = 0
    seen_item_ids: dict[str, str] = {}

    for panel in registry["panels"]:
        manifest, cases = load_panel(root / panel["path"])
        method = manifest["grading_method"]
        schema_path = schema_dir / SCHEMA_BY_METHOD[method]
        schema = json.loads(schema_path.read_text())
        validator = Draft202012Validator(schema)

        manifest_ids = set(manifest["case_ids"])
        file_ids = {case["case_id"] for case in cases}
        extra = file_ids - manifest_ids
        missing = sorted(manifest_ids - file_ids)
        if missing:
            errors.append(f"{panel['id']}: manifest missing files for {missing[:3]}{'...' if len(missing) > 3 else ''}")
        extra_cases = [case for case in cases if case["case_id"] in extra]
        bad_extra = [
            case["case_id"]
            for case in extra_cases
            if case.get("split") != "test"
        ]
        if bad_extra:
            errors.append(f"{panel['id']}: orphan case files {bad_extra[:3]}{'...' if len(bad_extra) > 3 else ''}")

        for case in cases:
            checked += 1
            for err in validator.iter_errors(case):
                loc = ".".join(str(p) for p in err.path) or "(root)"
                errors.append(f"{panel['id']}/{case['case_id']}: {loc}: {err.message}")
            prompt = str(case.get("prompt") or "")
            local: set[str] = set()
            for index, item in enumerate(case.get("rubric_items") or []):
                iid = str(item.get("item_id") or "")
                expected = item_id_for(prompt, index)
                if iid != expected:
                    errors.append(
                        f"{panel['id']}/{case['case_id']}: item_id {iid!r} "
                        f"!= md5(prompt)_{index} ({expected})"
                    )
                if iid in local:
                    errors.append(
                        f"{panel['id']}/{case['case_id']}: duplicate item_id {iid}"
                    )
                local.add(iid)
                prior = seen_item_ids.get(iid)
                if prior:
                    errors.append(
                        f"{panel['id']}/{case['case_id']}: item_id {iid} "
                        f"also used by {prior}"
                    )
                else:
                    seen_item_ids[iid] = f"{panel['id']}/{case['case_id']}"

    report = {
        "checked": checked,
        "panels": len(registry["panels"]),
        "errors": errors,
        "ok": not errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
