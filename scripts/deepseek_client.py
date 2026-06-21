"""Minimal robust DeepSeek V4 client for real LLM rollouts.

The API key is read ONLY from the DEEPSEEK_API_KEY environment variable; it is never
written to a file or printed. DeepSeek V4 is a reasoning model (it spends tokens on
hidden reasoning before the visible answer), so max_tokens must be generous and we read
`message.content` (the final answer) after the reasoning. Transient SSL/connection errors
are retried with backoff.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

try:
    import requests
except Exception as exc:  # pragma: no cover
    raise RuntimeError("the `requests` package is required for DeepSeek rollouts") from exc

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class DeepSeekError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise DeepSeekError("DEEPSEEK_API_KEY environment variable is not set")
    return key


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1500,
    temperature: float = 0.7,
    retries: int = 5,
    timeout: int = 120,
) -> str:
    """Return the assistant's final text content. Retries transient network errors."""
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=body, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise DeepSeekError(f"transient HTTP {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content") or ""
            if not content.strip():
                # All budget went to reasoning; retry once with a larger budget.
                body["max_tokens"] = min(int(body["max_tokens"] * 2), 4000)
                raise DeepSeekError("empty content (reasoning consumed budget)")
            return content
        except Exception as exc:  # noqa: BLE001 - broad to cover SSL/connection/json
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise DeepSeekError(f"DeepSeek call failed after {retries} retries: {last_err}")


def extract_json_array(text: str) -> list[Any]:
    """Pull the first JSON array out of an LLM response (tolerant of code fences/prose)."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("[")
        end = text.rfind("]")
        candidate = text[start : end + 1] if start >= 0 and end > start else None
    if candidate is None:
        return []
    try:
        value = json.loads(candidate)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def healthcheck() -> dict[str, Any]:
    reply = chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=64, temperature=0.0)
    return {"ok": "OK" in reply.upper(), "reply": reply[:50]}


if __name__ == "__main__":
    print(json.dumps(healthcheck(), ensure_ascii=False))
