"""Run the rule-based check on one description file."""

import argparse
import sys
from pathlib import Path

from checker.missing import check_missing


def print_report(result):
    print(f"있음 ({len(result['present'])}): {', '.join(result['present']) or '-'}")
    print(f"누락 ({len(result['missing'])}): {', '.join(result['missing']) or '-'}")
    print("부족한 부분 (gaps):")
    if result["gaps"]:
        for item, parts in result["gaps"].items():
            print(f"  {item}: {', '.join(parts)}")
    else:
        print("  -")
    print("판매자 질문:")
    if not result["questions"]:
        print("  -")
    for q in result["questions"]:
        print(f"  [{q['label']}] {q['question']}")
        print(f"    수정 예시: {q['fix_example']}")


def main():
    parser = argparse.ArgumentParser(description="Check a product description for missing information.")
    parser.add_argument("file", nargs="?", help="UTF-8 text file with one description")
    parser.add_argument("--garment", default="top", help="garment type in data/checklists.json (default: top)")
    parser.add_argument("--smoke", action="store_true", help="send one tool-call test request to OpenRouter")
    parser.add_argument("--llm", action="store_true", help="also run the model (at most 3 requests)")
    args = parser.parse_args()

    if args.smoke:
        from checker.llm import print_smoke_report, smoke_test

        report = smoke_test()
        print_smoke_report(report)
        return 0 if report["ok"] else 1

    if not args.file:
        parser.error("a file is required unless --smoke is used")

    try:
        text = Path(args.file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"Cannot read {args.file}: {error}", file=sys.stderr)
        return 1

    if not text.strip():
        print("입력이 비어 있습니다. 모든 항목을 누락으로 표시합니다.")

    try:
        result = check_missing(text, args.garment)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    print_report(result)
    if not args.llm:
        return 0

    from checker.llm import analyze, print_analysis

    print("\n모델 분석:")
    analysis = analyze(text, args.garment, result, Path(args.file).stem)
    print_analysis(analysis)
    return 0 if analysis["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
