#!/usr/bin/env python3
import json
from pathlib import Path
from load_panel import load_panel

root = Path(__file__).resolve().parents[1]
registry = json.loads((root / "registry.json").read_text())
published_target = registry.get("published_cases_per_panel", 25)
full_target = registry["cases_per_panel"]
errors = []
for panel in registry["panels"]:
    manifest, cases = load_panel(root / panel["path"])
    train = [case for case in cases if case.get("split") == "train"]
    test = [case for case in cases if case.get("split") == "test"]
    if len(manifest["case_ids"]) != published_target:
        errors.append(
            f"{panel['id']}: manifest has {len(manifest['case_ids'])}, "
            f"expected {published_target} published cases"
        )
    if {case["case_id"] for case in train} != set(manifest["case_ids"]):
        errors.append(f"{panel['id']}: manifest case_ids must be the train split")
    if len(train) != published_target:
        errors.append(f"{panel['id']}: expected {published_target} train cases, got {len(train)}")
    if test and len(test) != published_target:
        errors.append(f"{panel['id']}: expected {published_target} local test cases, got {len(test)}")
    if panel["n"] != full_target:
        errors.append(f"{panel['id']}: n {panel['n']} != target {full_target}")
print(json.dumps({"panels": len(registry["panels"]), "target": full_target, "published": published_target, "total": registry["totals"]["all_cases"], "errors": errors, "ok": not errors}, indent=2))
