"""Vague-quote scoring and rescoring of saved runs. No network request is ever sent."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate
from evaluate import load_latest_runs, match_quotes, score_vague


@pytest.mark.parametrize("expected, flagged", [
    ("여유 있는 핏", "여유 있는 핏"),
    ("여유 있는 핏", "여유있는 핏"),
    ("여유 있는 핏", "\"여유 있는 핏\"."),
    ("어깨 넉넉함", "넉넉함"),
    ("넉넉함", "어깨 넉넉함"),
    ("FREE", "FREE"),
    ("핏", "핏"),
])
def test_quotes_that_match(expected, flagged):
    assert match_quotes([expected], [flagged])["correct"] == [(expected, flagged)]


@pytest.mark.parametrize("expected, flagged", [
    ("여유 있는 핏", "핏"),
    ("가슴 60", "가슴 62"),
    ("여유 있는 핏", "부드럽고 여유 있는 핏이라 데일리로 좋아요"),
    ("여유 있는 핏", "!!"),
])
def test_quotes_that_do_not_match(expected, flagged):
    result = match_quotes([expected], [flagged])
    assert result["correct"] == []
    assert result["missed"] == [expected] and result["extra"] == [flagged]


def test_one_flagged_quote_cannot_satisfy_two_expected_quotes():
    result = match_quotes(["가슴 60", "가슴 60"], ["가슴 60"])
    assert len(result["correct"]) == 1 and result["missed"] == ["가슴 60"] and result["extra"] == []


def test_duplicate_flagged_quote_counts_as_extra():
    result = match_quotes(["가슴 60"], ["가슴 60", "가슴 60"])
    assert len(result["correct"]) == 1 and result["extra"] == ["가슴 60"]


def test_exact_match_wins_over_an_earlier_partial_match():
    result = match_quotes(["넉넉한 핏", "넉넉한 핏이에요"], ["넉넉한 핏이에요", "넉넉한 핏"])
    assert sorted(result["correct"]) == [("넉넉한 핏", "넉넉한 핏"), ("넉넉한 핏이에요", "넉넉한 핏이에요")]


def test_empty_key_and_no_flags():
    assert match_quotes([], []) == {"correct": [], "missed": [], "extra": []}


@pytest.mark.parametrize("analysis", [
    None,
    {"status": "failed", "items": None},
    {"status": "not_run", "items": None},
])
def test_a_run_without_a_report_is_not_scored_as_all_missed(analysis):
    assert score_vague({"vague_quotes": ["x"]}, analysis) is None


def save(folder, name, **fields):
    record = {"sample_id": "top_02", "saved_at": "2026-01-01T00:00:00", "status": "ok", "items": [], **fields}
    (folder / name).write_text(json.dumps(record), encoding="utf-8")


def test_latest_run_per_sample_is_used(tmp_path):
    save(tmp_path, "a.json", saved_at="2026-01-02T00:00:00", model="new")
    save(tmp_path, "b.json", saved_at="2026-01-01T00:00:00", model="old")
    runs, problems = load_latest_runs(tmp_path)
    assert runs["top_02"]["model"] == "new" and problems == []


def test_unreadable_run_is_reported(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    runs, problems = load_latest_runs(tmp_path)
    assert runs == {} and "bad.json" in problems[0]


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, "evaluate.py", *arguments],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )


def test_from_runs_is_labeled_and_prints_results(tmp_path):
    text = (ROOT / "data" / "samples" / "top_04.txt").read_text(encoding="utf-8")
    item = {"quote": "정핏", "type": "subjective", "question": "q", "fix_example": "핏: [확인 필요]"}
    save(
        tmp_path, "run.json", sample_id="top_04", model="saved/model", items=[item], requests_used=2,
        input_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    result = run_cli("--from-runs", str(tmp_path), "--sample", "top_04")
    assert "RESCORING SAVED RESPONSES" in result.stdout and "not a new check" in result.stdout
    assert "saved/model" in result.stdout
    assert "extra (1): 정핏" in result.stdout
    assert "the tool flagged 1 expression(s)" in result.stdout
    assert "[subjective]" in result.stdout
    assert "WARNING" not in result.stdout


def test_from_runs_warns_when_the_sample_changed(tmp_path):
    save(tmp_path, "run.json", sample_id="top_04", input_sha256="old")
    assert "has changed" in run_cli("--from-runs", str(tmp_path), "--sample", "top_04").stdout


def test_from_runs_handles_a_failed_run(tmp_path):
    save(tmp_path, "run.json", sample_id="top_04", status="failed", items=None, message="HTTP 429: raw")
    output = run_cli("--from-runs", str(tmp_path), "--sample", "top_04").stdout
    assert "not scored" in output and "HTTP 429: raw" in output


def test_from_runs_handles_no_saved_run(tmp_path):
    assert "No saved run for top_04" in run_cli("--from-runs", str(tmp_path), "--sample", "top_04").stdout


def test_llm_and_from_runs_cannot_be_combined():
    assert run_cli("--llm", "--from-runs").returncode == 2


def test_default_mode_stays_offline_and_shows_not_run():
    output = run_cli().stdout
    assert "5/5 samples passed" in output and "vague_quotes: not run" in output
