#!/usr/bin/env python3
import json, sys
from pathlib import Path

def load_panel(panel_dir: Path):
    manifest = json.loads((panel_dir / "manifest.json").read_text())
    cases_dir = panel_dir / "cases"
    cases = []
    seen = set()
    for cid in manifest["case_ids"]:
        path = cases_dir / f"{cid}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing published case: {path}")
        case = json.loads(path.read_text())
        cases.append(case)
        seen.add(cid)
    if cases_dir.is_dir():
        for path in sorted(cases_dir.glob("*.json")):
            case = json.loads(path.read_text())
            cid = case.get("case_id")
            if cid in seen:
                continue
            if case.get("split") == "test":
                cases.append(case)
                seen.add(cid)
    return manifest, cases

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    panel = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/drug_safety"
    manifest, cases = load_panel(panel)
    print(json.dumps({"benchmark": manifest["benchmark"], "n": len(cases)}, indent=2))
