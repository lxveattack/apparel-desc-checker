"""OpenRouter access. For now, only the one-request tool-call smoke test."""

import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 1024
TIMEOUT_SECONDS = 60

CHECK_MISSING_TOOL = {
    "type": "function",
    "function": {
        "name": "check_missing",
        "description": "Check which required items are missing from the product description.",
        "parameters": {
            "type": "object",
            "properties": {
                "garment_type": {"type": "string", "enum": ["top"]},
            },
            "required": ["garment_type"],
        },
    },
}


def mask_secrets(text, api_key):
    """Hide the API key and the OpenRouter user id. Use this on anything printed or saved."""
    text = text.replace(api_key, "[KEY]")
    text = re.sub(r'("user_id"\s*:\s*)"[^"]*"', r'\1"[USER_ID]"', text)
    return re.sub(r"\buser_[A-Za-z0-9]{8,}\b", "[USER_ID]", text)


def load_settings():
    """Returns (api_key, model, None), or (None, None, message) if a setting is empty."""
    load_dotenv(ENV_PATH)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "").strip()
    empty = [name for name, value in (("OPENROUTER_API_KEY", api_key), ("OPENROUTER_MODEL", model)) if not value]
    if empty:
        return None, None, f"Not set in .env: {', '.join(empty)}"
    return api_key, model, None


def smoke_test():
    """Send exactly one request, without retries, and report what came back.

    Nothing is executed for a returned tool call, so no second request can follow.
    """
    api_key, model, problem = load_settings()
    if problem:
        return {"ok": False, "requests_sent": 0, "message": problem}

    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "tools": [CHECK_MISSING_TOOL],
        "messages": [{
            "role": "user",
            "content": "Call the check_missing tool with garment_type set to top. This is a connection test.",
        }],
    }
    report = {"ok": False, "requests_sent": 1, "model": model}
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        report["message"] = f"Request failed: {type(error).__name__}: {mask_secrets(str(error), api_key)}"
        return report

    report["http_status"] = response.status_code
    report["raw_response"] = mask_secrets(response.text, api_key)
    try:
        body = response.json()
        choice = body["choices"][0]
        tool_calls = choice["message"].get("tool_calls") or []
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        report["message"] = "The response has no usable choices. See the raw response."
        return report

    report["finish_reason"] = choice.get("finish_reason")
    report["provider"] = body.get("provider")
    report["served_model"] = body.get("model")
    report["tool_calls"] = [call.get("function") for call in tool_calls]
    report["ok"] = any(call.get("function", {}).get("name") == "check_missing" for call in tool_calls)
    report["message"] = "The model called check_missing." if report["ok"] else "The model did not call check_missing."
    return report


def print_smoke_report(report):
    print(f"Requests sent: {report['requests_sent']}")
    for label, key in (
        ("Configured model", "model"), ("Served model", "served_model"), ("Provider", "provider"),
        ("HTTP status", "http_status"), ("Finish reason", "finish_reason"),
    ):
        if report.get(key) is not None:
            print(f"{label}: {report[key]}")
    if "tool_calls" in report:
        print(f"Tool calls: {json.dumps(report['tool_calls'], ensure_ascii=False)}")
    print(f"Result: {report['message']}")
    if "raw_response" in report:
        print("Raw response:")
        print(report["raw_response"])
