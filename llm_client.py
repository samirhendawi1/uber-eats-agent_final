"""Thin wrapper around the hosted qwen3 endpoint (OpenAI-compatible)."""

from __future__ import annotations
import json, re, requests
from config import LLM_BASE_URL, LLM_MODEL, LLM_API_KEY


def chat_completion(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """Send a chat-completion request and return the assistant text."""
    url = f"{LLM_BASE_URL}/v1/chat/completions"
    payload: dict = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    # Strip <think>...</think> blocks produced by reasoning-enabled models
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def chat_completion_json(messages: list[dict], temperature: float = 0.2) -> dict:
    """Return parsed JSON from the LLM (best-effort)."""
    raw = chat_completion(messages, temperature=temperature, json_mode=True)
    # Try to extract JSON from the response
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"error": "Failed to parse JSON", "raw": raw}
