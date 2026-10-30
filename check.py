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
    parser.add_argument("file", help="UTF-8 text file with one description")
    parser.add_argument("--garment", default="top", help="garment type in data/checklists.json (default: top)")
    args = parser.parse_args()

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
