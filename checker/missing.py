"""Rule-based presence check. All item words and templates live in data/checklists.json."""

import json
import re
import unicodedata
from pathlib import Path

CHECKLIST_PATH = Path(__file__).resolve().parent.parent / "data" / "checklists.json"

NUMBER = r"\d+(?:\.\d+)?(?:\s*[~/\-]\s*\d+(?:\.\d+)?)*"
CM_UNIT = r"(?:cm|센티|센치)"
LABEL_SEPARATOR = r"[\s:=\-은는이가약]*"
UNIT_DECLARATION = rf"\(\s*(?:단위\s*:?\s*)?{CM_UNIT}\s*\)|단위\s*:?\s*{CM_UNIT}"
# Words a size table line may hold besides label-number pairs.
TABLE_WORDS = r"실측|사이즈표|사이즈|치수|상세|(?<![a-z0-9])(?:xs|s|m|l|xl|xxl|2xl|3xl|free)(?![a-z0-9])"


def load_checklist(garment_type, path=CHECKLIST_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    garments = data["garments"]
    if garment_type not in garments:
        raise ValueError(f"Unknown garment type: {garment_type!r}. Available: {sorted(garments)}")
    return garments[garment_type], data["common"]


def _normalize(text):
    # NFKC folds full-width forms and the ㎝ sign, so the rules see one spelling.
    return unicodedata.normalize("NFKC", text or "").lower()


def _split_sentences(line):
    return [s.strip() for s in re.split(r"\.(?!\d)|[!?…]", line)]


def _sentences(text):
    return [s for line in text.split("\n") for s in _split_sentences(line) if s]


def _clauses(text):
    return [c.strip() for s in _sentences(text) for c in re.split(r",(?!\d)", s) if c.strip()]


def _has_value_after_label(clause, label, no_value_phrases):
    if label not in clause or any(p in clause for p in no_value_phrases):
        return False
    return bool(re.sub(r"[^\w]", "", clause.replace(label, "", 1)))


def _qualitative_present(item, text, no_value_phrases):
    # Values are matched per sentence so comma lists like "95, 100, 105" stay whole.
    # Labels are matched per clause so "소재: 추후 안내, 신축성 좋음" keeps the two apart.
    for sentence in _sentences(text):
        searchable = sentence
        if item.get("exclude_context"):
            searchable = re.sub(item["exclude_context"], " ", sentence)
        if any(_normalize(v) in searchable for v in item["values"]):
            return True
        if any(re.search(p, searchable) for p in item["patterns"]):
            return True
    return any(
        _has_value_after_label(clause, _normalize(label), no_value_phrases)
        for clause in _clauses(text)
        for label in item["labels"]
    )


def _alias_regex(part):
    aliases = sorted(part["aliases"], key=len, reverse=True)
    names = "|".join(re.escape(_normalize(a)) for a in aliases)
    # A body number such as "모델 어깨 40cm" belongs to the model, and "반소매" is not "소매".
    return rf"(?<![가-힣])(?<!모델 )(?:{names})"


def _is_table_line(segment, parts):
    """True if the segment holds only label-number pairs, a unit declaration and table words."""
    rest = re.sub(UNIT_DECLARATION, " ", segment)
    for part in parts.values():
        rest = re.sub(rf"{_alias_regex(part)}{LABEL_SEPARATOR}{NUMBER}(?:\s*{CM_UNIT})?", " ", rest)
    rest = re.sub(TABLE_WORDS, " ", rest)
    return not re.search(r"[가-힣a-z%]", rest)


def _has_bare_pair(segment, parts):
    return any(re.search(rf"{_alias_regex(p)}{LABEL_SEPARATOR}{NUMBER}", segment) for p in parts.values())


def _size_blocks(text, parts):
    """Runs of adjacent table lines. A blank line or any other sentence ends a run."""
    blocks, current = [], []
    for line in text.split("\n"):
        segments = [s for s in _split_sentences(line) if s]
        if not segments:
            blocks.append(current)
            current = []
        for segment in segments:
            is_row = _is_table_line(segment, parts) and (
                _has_bare_pair(segment, parts) or re.search(UNIT_DECLARATION, segment)
            )
            if is_row:
                current.append(segment)
            else:
                blocks.append(current)
                current = []
    blocks.append(current)
    return [block for block in blocks if block]


def _labeled_parts_with_number(item, text):
    parts = item["parts"]
    found = {
        key for key, part in parts.items()
        if re.search(rf"{_alias_regex(part)}{LABEL_SEPARATOR}{NUMBER}\s*{CM_UNIT}", text)
    }
    if item.get("shared_unit"):
        for block in _size_blocks(text, parts):
            joined = "\n".join(block)
            if re.search(UNIT_DECLARATION, joined):
                found |= {
                    key for key, part in parts.items()
                    if re.search(rf"{_alias_regex(part)}{LABEL_SEPARATOR}{NUMBER}", joined)
                }
    return found


def _context_parts_with_value(item, text):
    found = set()
    for sentence in _sentences(text):
        if re.search(item["context"], sentence):
            found |= {k for k, p in item["parts"].items() if re.search(p["regex"], sentence)}
    return found


def _numeric_status(item, text):
    """Returns (present, part keys without a value), both per the numeric-item rule."""
    if item["match"] == "labeled":
        found = _labeled_parts_with_number(item, text)
        present = bool(found)
    else:
        found = _context_parts_with_value(item, text)
        present = any(item["parts"][k].get("counts_for_presence", True) for k in found)
    return present, [k for k in item["parts"] if k not in found]


def check_missing(text, garment_type="top"):
    """Check one description. Vagueness is not judged here, only whether items are stated."""
    checklist, common = load_checklist(garment_type)
    normalized = _normalize(text)
    no_value_phrases = [_normalize(p) for p in common["no_value_phrases"]]

    result = {"present": [], "missing": [], "gaps": {}, "questions": []}
    for key, item in checklist["items"].items():
        if item["kind"] == "numeric":
            present, gap_parts = _numeric_status(item, normalized)
        else:
            present, gap_parts = _qualitative_present(item, normalized, no_value_phrases), []

        if not present:
            result["missing"].append(key)
            result["questions"].append({
                "item": key, "part": None, "label": item["label"],
                "question": item["question"], "fix_example": item["fix_example"],
            })
            continue

        result["present"].append(key)
        if gap_parts:
            result["gaps"][key] = gap_parts
            for part_key in gap_parts:
                part = item["parts"][part_key]
                result["questions"].append({
                    "item": key, "part": part_key, "label": f"{item['label']} - {part['label']}",
                    "question": part["question"], "fix_example": part["fix_example"],
                })
    return result
