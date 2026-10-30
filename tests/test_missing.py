"""Rule-based presence check only. No model is involved."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checker.missing import check_missing, load_checklist

ALL_ITEMS = {
    "material", "sizes", "measurements", "model_info", "colors", "thickness",
    "stretch", "see_through", "fit", "care", "origin",
}


def is_missing(text, item):
    return item in check_missing(text)["missing"]


# Numeric items: measurements

def test_label_number_and_unit_together_count():
    result = check_missing("총장 65cm")
    assert "measurements" in result["present"]
    assert result["gaps"]["measurements"] == ["shoulder", "chest", "sleeve"]


def test_partial_measurements_list_gaps():
    assert check_missing("총장 68cm, 가슴단면 55cm")["gaps"]["measurements"] == ["shoulder", "sleeve"]


def test_wording_only_measurements_are_missing_without_gaps():
    result = check_missing("어깨 넉넉하고 기장 적당해요")
    assert "measurements" in result["missing"]
    assert "measurements" not in result["gaps"]


def test_vague_part_goes_to_gaps_when_another_part_has_a_number():
    assert "shoulder" in check_missing("어깨 넉넉함, 총장 65cm")["gaps"]["measurements"]


def test_number_without_unit_does_not_count():
    assert is_missing("가슴 55", "measurements")


def test_number_for_another_item_does_not_count_for_measurements():
    assert is_missing("모델 키 165cm", "measurements")


def test_model_body_part_does_not_count_for_measurements():
    assert is_missing("모델 어깨 40cm", "measurements")


def test_decimal_and_full_width_unit():
    assert not is_missing("총장 65.5㎝", "measurements")


def test_details_in_image_are_missing():
    assert is_missing("실측은 상세 이미지 참고", "measurements")


# Shared unit declarations

@pytest.mark.parametrize("text", [
    "어깨 48 가슴 55 총장 70 소매 22 (cm)",
    "어깨 48 가슴 55 총장 70 소매 22 (단위: cm)",
    "(단위: cm)\n어깨 48\n가슴 55\n총장 70\n소매 22",
    "실측 사이즈 (cm)\nM 어깨 48 가슴 55 총장 70 소매 22",
    "편하게 입기 좋아요.\n어깨 48 가슴 55 총장 70 소매 22 (cm)",
])
def test_shared_unit_inside_size_block(text):
    result = check_missing(text)
    assert "measurements" in result["present"]
    assert "measurements" not in result["gaps"]


@pytest.mark.parametrize("text", [
    "어깨 48 가슴 55\n\n소매 끝단 2.5 (cm)만 수선 가능",
    "어깨 48 가슴 55\n\n(cm)",
    "어깨 48 가슴 55\n배송은 3일 걸려요\n(단위: cm)",
    "가슴 55라서 편해요 (cm)",
])
def test_shared_unit_outside_size_block_does_not_apply(text):
    assert is_missing(text, "measurements")


# Numeric items: model_info

def test_size_worn_alone_is_missing():
    result = check_missing("모델 M 착용")
    assert "model_info" in result["missing"]
    assert "model_info" not in result["gaps"]


def test_height_and_size_worn_leave_weight_as_gap():
    assert check_missing("모델 168cm, M 착용")["gaps"]["model_info"] == ["weight"]


def test_complete_model_info_has_no_gaps():
    result = check_missing("모델 키 168cm, 몸무게 50kg, FREE 착용")
    assert "model_info" in result["present"]
    assert "model_info" not in result["gaps"]


def test_model_number_without_unit_is_missing():
    assert is_missing("모델 165", "model_info")


def test_weight_counts_for_presence_and_size_worn_is_a_gap():
    assert check_missing("모델 몸무게 50kg")["gaps"]["model_info"] == ["height", "size_worn"]


def test_measurement_is_not_model_height():
    assert is_missing("총장 165cm", "model_info")


# Qualitative items

@pytest.mark.parametrize("text", ["소재: 추후 안내", "소재:", "소재"])
def test_label_with_no_value_is_missing(text):
    assert is_missing(text, "material")


def test_no_value_phrase_stays_in_its_own_clause():
    result = check_missing("소재: 추후 안내, 신축성 좋음")
    assert "material" in result["missing"]
    assert "stretch" in result["present"]


def test_clear_negative_is_present():
    assert not is_missing("신축성 없음", "stretch")


def test_synonym_is_present():
    assert not is_missing("코튼 100%", "material")


@pytest.mark.parametrize("text", ["면 100%", "면 소재", "부드러운 니트"])
def test_material_words_are_present(text):
    assert not is_missing(text, "material")


@pytest.mark.parametrize("text", ["가슴단면 55cm", "표면이 매끄러워요", "앞면 프린트"])
def test_syllable_inside_another_word_is_not_material(text):
    assert is_missing(text, "material")


def test_vague_word_is_present_for_qualitative_item():
    assert not is_missing("여유 있는 핏", "fit")


def test_spandex_in_blend_is_not_stretch():
    assert is_missing("폴리 95% 스판덱스 5%", "stretch")


# Sizes

@pytest.mark.parametrize("text", ["S/M/L", "FREE", "사이즈: 95, 100, 105"])
def test_size_labels_are_present(text):
    assert not is_missing(text, "sizes")


def test_size_worn_by_model_is_not_an_available_size():
    assert is_missing("모델 M 착용", "sizes")


def test_available_sizes_next_to_size_worn_still_count():
    assert not is_missing("사이즈 S/M/L 모델 M 착용", "sizes")


def test_size_chart_in_image_is_missing():
    assert is_missing("사이즈표는 상세 이미지 참고", "sizes")


# Empty or unrelated input

@pytest.mark.parametrize("text", ["", "   \n\n ", "오늘 배송 빠르게 보내드려요"])
def test_empty_or_unrelated_input(text):
    result = check_missing(text)
    assert set(result["missing"]) == ALL_ITEMS
    assert result["present"] == []
    assert result["gaps"] == {}


# Output shape and data file

def test_every_missing_item_and_gap_gets_a_question_with_a_placeholder():
    result = check_missing("모델 168cm, M 착용\n총장 65cm")
    expected = len(result["missing"]) + sum(len(parts) for parts in result["gaps"].values())
    assert len(result["questions"]) == expected
    assert all(q["question"] and "[확인 필요]" in q["fix_example"] for q in result["questions"])


def test_checklist_has_the_eleven_agreed_keys():
    checklist, _ = load_checklist("top")
    assert set(checklist["items"]) == ALL_ITEMS


def test_unknown_garment_type_is_an_error():
    with pytest.raises(ValueError):
        check_missing("총장 65cm", "pants")
