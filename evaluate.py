"""Score the checker against data/expected.json.

Missing items and gaps come from the rules. Vague quotes come from a new model run (--llm)
or from saved model responses (--from-runs). Without either flag, no model request is sent.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

from checker.llm import RUNS_DIR, analyze
from checker.missing import check_missing, load_checklist

ROOT = Path(__file__).resolve().parent
EXPECTED_PATH = ROOT / "data" / "expected.json"
SAMPLES_DIR = ROOT / "data" / "samples"
PAUSE_SECONDS = 10
MIN_PARTIAL_LENGTH = 3
MIN_PARTIAL_RATIO = 0.5


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
    return {"passed": passed, "missing": missing, "gaps": gaps, "rule_result": result}


def normalize_quote(quote):
    """NFC, no whitespace, no surrounding quotes or punctuation."""
    without_space = re.sub(r"\s+", "", unicodedata.normalize("NFC", quote))
    return re.sub(r"^[\W_]+|[\W_]+$", "", without_space)


def quote_match_rank(expected, flagged):
    """Returns (is_exact, length_ratio) for a pair that can match, otherwise None.

    Identical quotes match at any length. Otherwise one must contain the other, and the
    shorter must be long enough that "핏" cannot match "여유 있는 핏".
    """
    a, b = normalize_quote(expected), normalize_quote(flagged)
    if not a or not b:
        return None
    if a == b:
        return (1, 1.0)
    shorter, longer = sorted((a, b), key=len)
    ratio = len(shorter) / len(longer)
    if shorter in longer and len(shorter) >= MIN_PARTIAL_LENGTH and ratio >= MIN_PARTIAL_RATIO:
        return (0, ratio)
    return None


def match_quotes(expected, flagged):
    """Match one to one: exact first, then highest ratio, then key order, then flagged order."""
    candidates = []
    for i, expected_quote in enumerate(expected):
        for j, flagged_quote in enumerate(flagged):
            rank = quote_match_rank(expected_quote, flagged_quote)
            if rank:
                candidates.append((-rank[0], -rank[1], i, j))
    used_expected, used_flagged, correct = set(), set(), []
    for _, _, i, j in sorted(candidates):
        if i not in used_expected and j not in used_flagged:
            used_expected.add(i)
            used_flagged.add(j)
            correct.append((expected[i], flagged[j]))
    return {
        "correct": correct,
        "missed": [q for i, q in enumerate(expected) if i not in used_expected],
        "extra": [q for j, q in enumerate(flagged) if j not in used_flagged],
    }


def load_latest_runs(runs_dir):
    """The newest saved run per sample. Unreadable files are reported, not silently skipped."""
    latest, problems = {}, []
    for path in sorted(Path(runs_dir).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            sample_id, saved_at = record["sample_id"], record["saved_at"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        if sample_id not in latest or saved_at >= latest[sample_id]["saved_at"]:
            latest[sample_id] = record
    return latest, problems


def score_vague(entry, analysis):
    """Returns the quote comparison, or None if the model did not produce a report."""
    if analysis is None or analysis.get("status") != "ok":
        return None
    return match_quotes(entry["vague_quotes"], [item["quote"] for item in analysis.get("items") or []])


def print_comparison(name, comparison):
    print(f"  {name}:")
    for key in ("correct", "missed", "extra"):
        print(f"    {key} ({len(comparison[key])}): {', '.join(comparison[key]) or '-'}")


def print_vague(entry, analysis, vague):
    if analysis is None:
        print(f"  vague_quotes: not run ({len(entry['vague_quotes'])} expected)")
        return
    print(f"  model: status={analysis.get('status')} requests={analysis.get('requests_used')}")
    if vague is None:
        print(f"  vague_quotes: not scored. {analysis.get('message')}")
        return
    print("  vague_quotes:")
    pairs = [q if q == f else f"{q} <= {f}" for q, f in vague["correct"]]
    print(f"    correct ({len(pairs)}): {' | '.join(pairs) or '-'}")
    print(f"    missed ({len(vague['missed'])}): {' | '.join(vague['missed']) or '-'}")
    print(f"    extra ({len(vague['extra'])}): {' | '.join(vague['extra']) or '-'}")
    print("  flagged, for review (type is not scored):")
    for item in analysis.get("items") or []:
        print(f"    \"{item['quote']}\" [{item['type']}]")
        print(f"      question: {item['question']}")
        print(f"      fix: {item['fix_example']}")
        if item.get("new_numbers"):
            print(f"      WARNING number not in the text: {', '.join(item['new_numbers'])}")
    if not analysis.get("items"):
        print("    -")
    for item in analysis.get("rejected_items") or []:
        print(f"    REJECTED quote not in the text: \"{item.get('quote')}\"")
    if entry.get("nearly_complete"):
        print(f"  nearly_complete: the tool flagged {len(analysis.get('items') or [])} expression(s) as vague")


def print_sample(sample_id, entry, score, analysis, vague):
    print(f"[{'PASS' if score['passed'] else 'FAIL'}] {sample_id}")
    print_comparison("missing", score["missing"])
    print_comparison("gaps", score["gaps"])
    print_vague(entry, analysis, vague)
    if entry.get("limitation_note"):
        print(f"  limitation_note: {entry['limitation_note']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Score the checker against the answer key.")
    parser.add_argument("--sample", help="score one sample, for example top_03")
    parser.add_argument("--llm", action="store_true", help="run the model on each sample (at most 3 requests each)")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS, help="seconds between samples with --llm")
    parser.add_argument(
        "--from-runs", nargs="?", const=str(RUNS_DIR), metavar="DIR",
        help="rescore saved model responses (default folder: runs/). Sends no request",
    )
    args = parser.parse_args()
    if args.llm and args.from_runs:
        parser.error("use either --llm or --from-runs, not both")

    try:
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"Cannot read {EXPECTED_PATH}: {error}", file=sys.stderr)
        return 1

    if args.sample and args.sample not in expected:
        print(f"Unknown sample: {args.sample}. Available: {', '.join(sorted(expected))}", file=sys.stderr)
        return 1

    saved_runs = {}
    if args.from_runs:
        print("=== RESCORING SAVED RESPONSES: no model request is sent. ===")
        print("=== This is not a new check of the current model. ===\n")
        saved_runs, problems = load_latest_runs(args.from_runs)
        for problem in problems:
            print(f"Skipped unreadable run: {problem}")

    sample_ids = [args.sample] if args.sample else sorted(expected)
    passed = 0
    requests_used = 0
    totals = {"correct": 0, "missed": 0, "extra": 0}
    scored = 0
    for position, sample_id in enumerate(sample_ids):
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

        analysis = None
        if args.llm:
            if position:
                time.sleep(args.pause)
            analysis = analyze(text, entry["garment_type"], score["rule_result"], sample_id)
            requests_used += analysis["requests_used"]
        elif args.from_runs:
            analysis = saved_runs.get(sample_id)
            if analysis is None:
                print(f"No saved run for {sample_id}.")
            else:
                print(f"Saved run for {sample_id}: {analysis['saved_at']}, model {analysis.get('model')}")
                if analysis.get("input_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
                    print("WARNING: the sample has changed since this run was saved.")

        vague = score_vague(entry, analysis)
        if vague:
            scored += 1
            for key in totals:
                totals[key] += len(vague[key])
        print_sample(sample_id, entry, score, analysis, vague)
        passed += score["passed"]

    print(f"Missing and gaps: {passed}/{len(sample_ids)} samples passed")
    if args.llm or args.from_runs:
        print(
            f"Vague quotes on {scored}/{len(sample_ids)} scored samples: "
            f"correct {totals['correct']}, missed {totals['missed']}, extra {totals['extra']}"
        )
    if args.llm:
        print(f"Model requests sent: {requests_used}")
    return 0 if passed == len(sample_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
