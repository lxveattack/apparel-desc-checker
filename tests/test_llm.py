"""The model loop and page with fake responses. No network request is ever sent."""

import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app
from checker import llm
from checker.missing import check_missing

TEXT = "코튼 100% 티셔츠입니다. 총장 68cm"
USER_ID = "user_3FakeUserIdForTest"
GOOD_ITEM = {"quote": "코튼", "type": "incomplete_spec", "question": "혼용률은요?", "fix_example": "코튼 [확인 필요]%"}


def http(status, text):
    response = requests.Response()
    response.status_code = status
    response._content = text.encode("utf-8")
    return response


def reply(*calls, finish="tool_calls", content=None):
    tool_calls = [
        {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": arguments}}
        for i, (name, arguments) in enumerate(calls)
    ]
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body = {
        "model": "fake/model", "user_id": USER_ID,
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 15}},
        "choices": [{"finish_reason": finish, "message": message}],
    }
    return http(200, json.dumps(body))


CHECK = ("check_missing", '{"garment_type": "top"}')
REPORT = ("report_vague_expressions", json.dumps({"items": [GOOD_ITEM]}, ensure_ascii=False))
NO_TOOL = {"finish": "stop", "content": "plain text"}


class FakeModel:
    """Plays back responses in order. An exception in the list is raised instead."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def run(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "FAKEKEY")
    monkeypatch.setenv("OPENROUTER_MODEL", "fake/model")
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path)

    def _run(*responses, text=TEXT):
        fake = FakeModel(*responses)
        monkeypatch.setattr(llm.requests, "post", fake.post)
        analysis = llm.analyze(text, "top", check_missing(text), "test")
        analysis["fake"] = fake
        return analysis

    return _run


# Normal flow

def test_normal_flow_uses_two_requests_and_stops_after_a_valid_report(run):
    result = run(reply(CHECK), reply(REPORT))
    assert result["status"] == "ok"
    assert result["requests_used"] == 2
    assert result["items"] == [{**GOOD_ITEM, "new_numbers": []}]
    assert [c["name"] for s in result["trace"] for c in s["tool_calls"]] == ["check_missing", "report_vague_expressions"]


def test_empty_report_is_valid(run):
    result = run(reply(CHECK), reply(("report_vague_expressions", '{"items": []}')))
    assert result["status"] == "ok" and result["items"] == []


def test_valid_report_without_check_missing_is_accepted_and_noted(run):
    result = run(reply(REPORT))
    assert result["status"] == "ok" and result["requests_used"] == 1
    assert "skipped check_missing" in result["message"]


def test_checklist_is_in_the_prompt_and_there_is_no_get_checklist_tool(run):
    result = run(reply(CHECK), reply(REPORT))
    first = result["fake"].calls[0]
    assert "see_through" in first["messages"][0]["content"]
    assert [t["function"]["name"] for t in first["tools"]] == ["check_missing", "report_vague_expressions"]
    assert first["max_tokens"] == llm.MAX_TOKENS


def test_check_missing_returns_the_rule_result_for_the_original_text(run):
    result = run(reply(("check_missing", '{"garment_type": "top", "text": "invented"}')), reply(REPORT))
    tool_message = next(m for m in result["fake"].calls[1]["messages"] if m["role"] == "tool")
    assert json.loads(tool_message["content"])["missing"] == check_missing(TEXT)["missing"]


# Retries and the request limit

def test_no_tool_call_retries_within_the_limit(run):
    result = run(reply(**NO_TOOL), reply(CHECK), reply(REPORT))
    assert result["status"] == "ok" and result["requests_used"] == 3


def test_no_tool_call_three_times_fails_at_the_limit(run):
    result = run(reply(**NO_TOOL), reply(**NO_TOOL), reply(**NO_TOOL), reply(REPORT))
    assert result["status"] == "failed" and result["requests_used"] == 3
    assert len(result["fake"].calls) == 3
    assert "did not call a tool" in result["message"]


@pytest.mark.parametrize("arguments", [
    "{not json",
    '{"items": "x"}',
    '{"items": [{"quote": "코튼"}]}',
    json.dumps({"items": [{**GOOD_ITEM, "type": "made_up"}]}),
    json.dumps({"items": [{**GOOD_ITEM, "quote": " "}]}),
])
def test_invalid_report_retries_and_never_becomes_a_result(run, arguments):
    bad = ("report_vague_expressions", arguments)
    result = run(reply(bad), reply(bad), reply(bad))
    assert result["status"] == "failed" and result["requests_used"] == 3
    assert result["items"] is None
    assert "invalid" in result["message"]


def test_invalid_report_can_recover(run):
    result = run(reply(CHECK), reply(("report_vague_expressions", "{not json")), reply(REPORT))
    assert result["status"] == "ok" and result["requests_used"] == 3


def test_malformed_check_missing_arguments_still_get_the_rule_result(run):
    result = run(reply(("check_missing", "{not json")), reply(REPORT))
    assert result["status"] == "ok"
    assert "arguments ignored" in result["trace"][0]["tool_calls"][0]["result"]
    tool_message = next(m for m in result["fake"].calls[1]["messages"] if m["role"] == "tool")
    assert "missing" in json.loads(tool_message["content"])


def test_repeating_check_missing_cannot_pass_the_limit(run):
    result = run(reply(CHECK), reply(CHECK), reply(CHECK), reply(REPORT))
    assert result["status"] == "failed" and len(result["fake"].calls) == 3
    assert "repeated call" in result["trace"][1]["tool_calls"][0]["result"]


def test_unknown_tool_is_rejected(run):
    result = run(reply(("get_checklist", "{}")), reply(CHECK), reply(REPORT))
    assert result["status"] == "ok" and "unknown tool" in result["trace"][0]["tool_calls"][0]["result"]


# Failures that stop at once, with no retry

def test_length_is_a_clear_failure_even_with_a_tool_call(run):
    result = run(reply(REPORT, finish="length"), reply(REPORT))
    assert result["status"] == "failed" and result["requests_used"] == 1
    assert result["items"] is None
    assert "finish_reason: length" in result["message"]
    assert "reasoning_tokens" in result["message"]


@pytest.mark.parametrize("status", [429, 401, 404, 500])
def test_http_error_is_shown_as_returned_without_retry(run, status):
    raw = f'{{"error":{{"code":{status},"message":"raw error text"}}}}'
    result = run(http(status, raw), reply(REPORT))
    assert result["status"] == "failed" and len(result["fake"].calls) == 1
    assert raw in result["message"]


@pytest.mark.parametrize("response", [
    http(200, "<html>not json</html>"),
    http(200, '{"error":{"code":502,"message":"upstream"}}'),
    http(200, '{"choices":[]}'),
    http(200, '{"choices":[{"message":null}]}'),
])
def test_unusable_200_response_fails_without_retry(run, response):
    result = run(response, reply(REPORT))
    assert result["status"] == "failed" and len(result["fake"].calls) == 1
    assert "no usable choices" in result["message"]


@pytest.mark.parametrize("error", [requests.ConnectionError("down"), requests.Timeout("slow")])
def test_network_failure_fails_without_retry(run, error):
    result = run(error, reply(REPORT))
    assert result["status"] == "failed" and len(result["fake"].calls) == 1


# Guards

def report_of(*items):
    return reply(("report_vague_expressions", json.dumps({"items": list(items)}, ensure_ascii=False)))


def test_quote_not_in_the_text_is_rejected_and_not_a_result(run):
    result = run(report_of(GOOD_ITEM, {**GOOD_ITEM, "quote": "최고급 실크"}))
    assert [i["quote"] for i in result["items"]] == ["코튼"]
    assert [i["quote"] for i in result["rejected_items"]] == ["최고급 실크"]
    assert "Rejected 1 quote" in result["message"]


def test_guard_accepts_a_quote_with_different_line_breaks(run):
    result = run(report_of({**GOOD_ITEM, "quote": "입니다. 총장"}), text="코튼 티셔츠입니다.\r\n총장 68cm")
    assert len(result["items"]) == 1 and result["rejected_items"] == []


def test_guard_does_not_accept_a_partial_word_change(run):
    result = run(report_of({**GOOD_ITEM, "quote": "코튼 90%"}))
    assert result["items"] == [] and len(result["rejected_items"]) == 1


@pytest.mark.parametrize("fix, marked", [
    ("코튼 [확인 필요]%", []),
    ("코튼 100%, 총장 68cm", []),
    ("코튼 95%", ["95"]),
    ("코튼 100%, 어깨 45.5cm", ["45.5"]),
    ("총장 6cm", ["6"]),
])
def test_new_number_in_a_fix_example_is_marked(run, fix, marked):
    result = run(report_of({**GOOD_ITEM, "fix_example": fix}))
    assert result["items"][0]["new_numbers"] == marked


def test_guard_results_are_saved(run):
    result = run(report_of({**GOOD_ITEM, "fix_example": "코튼 95%"}, {**GOOD_ITEM, "quote": "없는 말"}))
    record = json.loads(saved_run(result))
    assert record["items"][0]["new_numbers"] == ["95"]
    assert record["rejected_items"][0]["quote"] == "없는 말"


# Not run

def test_empty_input_sends_no_request(run):
    result = run(reply(REPORT), text="  \n ")
    assert result["status"] == "not_run" and result["fake"].calls == []


def test_missing_settings_send_no_request(run, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "")
    result = run(reply(REPORT))
    assert result["status"] == "not_run" and result["fake"].calls == []
    assert "OPENROUTER_MODEL" in result["message"]


# Saved runs

def saved_run(result):
    return Path(result["run_file"]).read_text(encoding="utf-8")


def test_run_is_saved_with_the_trace_and_masked_secrets(run):
    result = run(reply(CHECK), reply(REPORT))
    saved = saved_run(result)
    record = json.loads(saved)
    assert record["status"] == "ok" and record["requests_used"] == 2
    assert record["model"] == "fake/model" and len(record["input_sha256"]) == 64
    assert len(record["exchanges"]) == 2 and record["trace"]
    assert "FAKEKEY" not in saved and USER_ID not in saved


def test_failed_run_is_saved_and_masks_secrets_in_raw_errors(run):
    result = run(http(429, f'{{"error":{{"message":"limit FAKEKEY {USER_ID}"}},"user_id":"{USER_ID}"}}'))
    saved = saved_run(result)
    assert json.loads(saved)["status"] == "failed"
    for text in (saved, result["message"]):
        assert "FAKEKEY" not in text and USER_ID not in text


def test_run_that_cannot_be_saved_still_returns_a_result(run, monkeypatch, tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("x")
    monkeypatch.setattr(llm, "RUNS_DIR", blocked)
    result = run(reply(REPORT))
    assert result["status"] == "ok" and result["run_file"] is None
    assert "could not be saved" in result["message"]


# The page

def post_page(monkeypatch, fake, action):
    monkeypatch.setattr(llm.requests, "post", fake.post)
    response = app.test_client().post("/", data={"description": TEXT, "action": action})
    return response.status_code, response.get_data(as_text=True)


@pytest.mark.parametrize("responses, expected", [
    ((http(429, '{"error":{"message":"raw limit text"}}'),), "raw limit text"),
    ((reply(REPORT, finish="length"),), "finish_reason: length"),
    ((reply(**NO_TOOL), reply(**NO_TOOL), reply(**NO_TOOL)), "did not call a tool"),
    ((requests.ConnectionError("down"),), "ConnectionError"),
])
def test_page_keeps_rule_results_when_the_model_fails(run, monkeypatch, responses, expected):
    status, page = post_page(monkeypatch, FakeModel(*responses), "model")
    assert status == 200
    assert '<span class="status">failed</span>' in page
    assert expected in page
    assert "see_through" in page and "measurements.shoulder" in page and "두께감을 알려주세요" in page
    assert "법적 적합성" in page


def test_page_shows_the_trace_when_the_model_succeeds(run, monkeypatch):
    status, page = post_page(monkeypatch, FakeModel(reply(CHECK), reply(REPORT)), "model")
    assert status == 200
    assert '<span class="status">ok</span>' in page
    assert "check_missing" in page and "report_vague_expressions" in page
    assert "see_through" in page


def test_page_shows_the_vague_table_with_found_and_fill_kept_apart(run, monkeypatch):
    item = {"quote": "코튼", "type": "incomplete_spec", "question": "혼용률은요?", "fix_example": "코튼 95%"}
    _, page = post_page(monkeypatch, FakeModel(report_of(item, {**item, "quote": "지어낸 말"})), "model")
    assert "본문에서 찾은 표현" in page and "판매자가 채울 내용" in page
    assert '<td class="found">코튼</td>' in page and '<td class="found">incomplete_spec</td>' in page
    assert "혼용률은요?" in page and "코튼 95%" in page
    assert "본문에 없는 숫자: 95" in page
    assert "제외한 인용" in page and "<li>지어낸 말</li>" in page


def test_page_with_an_empty_report_says_so(run, monkeypatch):
    _, page = post_page(monkeypatch, FakeModel(report_of()), "model")
    assert "찾은 표현 없음" in page


def test_page_escapes_model_text(run, monkeypatch):
    item = {**GOOD_ITEM, "question": "<script>alert(1)</script>"}
    _, page = post_page(monkeypatch, FakeModel(report_of(item)), "model")
    assert "<script>alert" not in page


def test_rule_only_button_sends_no_request(run, monkeypatch):
    fake = FakeModel()
    status, page = post_page(monkeypatch, fake, "rules")
    assert status == 200 and fake.calls == []
    assert '<span class="status">not_run</span>' in page and "see_through" in page


def test_page_with_no_settings_shows_not_run(run, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    fake = FakeModel()
    _, page = post_page(monkeypatch, fake, "model")
    assert fake.calls == []
    assert '<span class="status">not_run</span>' in page and "OPENROUTER_API_KEY" in page
