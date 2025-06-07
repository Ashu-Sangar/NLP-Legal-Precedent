import os
import json
import argparse
from pathlib import Path
from typing import List, Union

def get_base_dir() -> Path:
    env = os.getenv("CASE_JSON_DIR")
    if env:
        return Path(env)
    project_root = Path(__file__).resolve().parents[1]
    default = project_root / "data" / "Caselaw_Pennsylvania_State_Reports_1845-2017"
    if not default.exists():
        raise FileNotFoundError(
            f"Could not find default data directory at {default}. "
            "Set CASE_JSON_DIR or pass --json-dir explicitly."
        )
    return default

def get_gold_labels(
    case_id: Union[int, str],
    base_dir: Path = None,
) -> List[str]:
    if base_dir is None:
        base_dir = get_base_dir()

    data = None
    # only case_id lookup
    for json_file in base_dir.rglob("**/json/*.json"):
        with open(json_file, "r") as f:
            d = json.load(f)
            if str(d.get("id")) == str(case_id):
                data = d
                break
    if data is None:
        raise FileNotFoundError(
            f"Case ID {case_id} not found under {base_dir}"
        )

    # Extract numeric case_ids
    results = []
    for entry in data.get("cites_to", []):
        for cid in entry.get("case_ids", []):
            results.append(str(cid))

    return sorted(set(results))

def parse_args():
    p = argparse.ArgumentParser(
        description="Retrieve gold labels (cited precedents) from case JSON files"
    )
    p.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Root directory containing nested '*/json/' folders with case JSON files"
    )
    p.add_argument(
        "--case-id",
        type=str,
        required=True,
        help="Numeric case ID (matches the 'id' field in JSON)"
    )
    return p.parse_args()

def main():
    args = parse_args()
    gold = get_gold_labels(
        case_id=args.case_id,
        base_dir=args.json_dir,
    )
    print("Gold labels:")
    for label in gold:
        print(label)

if __name__ == "__main__":
    main()