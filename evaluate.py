"""Score the rule-based check against data/expected.json. No model request is sent."""

import argparse
import json
import sys
from pathlib import Path

from checker.missing import check_missing, load_checklist

ROOT = Path(__file__).resolve().parent
EXPECTED_PATH = ROOT / "data" / "expected.json"
SAMPLES_DIR = ROOT / "data" / "samples"


def compare(expected, found):
    expected, found = set(expected), set(found)
    return {
        "correct": sorted(expected & found),
        "missed": sorted(expected - found),
        "extra": sorted(found - expected),
    }


def gap_pairs(gaps):
    return [f"{item}.{part}" for item, parts in gaps.items() for part in parts]


def key_problems(entry):
    """Keys the checklist does not know, which would otherwise look like plain mismatches."""
    items = load_checklist(entry["garment_type"])[0]["items"]
    problems = [f"unknown item in missing: {key}" for key in entry["missing"] if key not in items]
    for item, parts in entry["gaps"].items():
        if item not in items:
            problems.append(f"unknown item in gaps: {item}")
            continue
        known_parts = items[item].get("parts", {})
        problems += [f"unknown part in gaps: {item}.{part}" for part in parts if part not in known_parts]
    return problems


def score_sample(entry, text):
    result = check_missing(text, entry["garment_type"])
    missing = compare(entry["missing"], result["missing"])
    gaps = compare(gap_pairs(entry["gaps"]), gap_pairs(result["gaps"]))
    passed = not (missing["missed"] or missing["extra"] or gaps["missed"] or gaps["extra"])
    return {"passed": passed, "missing": missing, "gaps": gaps}


def print_comparison(name, comparison):
    print(f"  {name}:")
    for key in ("correct", "missed", "extra"):
        print(f"    {key} ({len(comparison[key])}): {', '.join(comparison[key]) or '-'}")


def print_sample(sample_id, entry, score):
    print(f"[{'PASS' if score['passed'] else 'FAIL'}] {sample_id}")
    print_comparison("missing", score["missing"])
    print_comparison("gaps", score["gaps"])
    print(f"  vague_quotes: not run in this step ({len(entry['vague_quotes'])} expected)")
    if entry.get("limitation_note"):
        print(f"  limitation_note: {entry['limitation_note']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Score missing items and gaps against the answer key.")
    parser.add_argument("--sample", help="score one sample, for example top_03")
    args = parser.parse_args()

    try:
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"Cannot read {EXPECTED_PATH}: {error}", file=sys.stderr)
        return 1

    if args.sample and args.sample not in expected:
        print(f"Unknown sample: {args.sample}. Available: {', '.join(sorted(expected))}", file=sys.stderr)
        return 1

    sample_ids = [args.sample] if args.sample else sorted(expected)
    passed = 0
    for sample_id in sample_ids:
        entry = expected[sample_id]
        path = SAMPLES_DIR / f"{sample_id}.txt"
        try:
            text = path.read_text(encoding="utf-8")
            problems = key_problems(entry)
            if problems:
                raise ValueError("; ".join(problems))
            score = score_sample(entry, text)
        except (OSError, ValueError, KeyError) as error:
            print(f"[ERROR] {sample_id}: {type(error).__name__}: {error}\n")
            continue
        print_sample(sample_id, entry, score)
        passed += score["passed"]

    print(f"Missing and gaps: {passed}/{len(sample_ids)} samples passed")
    return 0 if passed == len(sample_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
