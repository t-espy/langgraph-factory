"""LLM client wrappers for Docker Model Runner."""

import json
import sys
import threading
import time
from dataclasses import dataclass, field

import requests
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

from langgraph_factory.config import DMR_BASE_URL, DMR_API_KEY
from langgraph_factory.utils import extract_json

_print_lock = threading.Lock()


@dataclass
class LLMStats:
    """Statistics from a single LLM call."""
    model: str = ""
    label: str = ""
    elapsed_s: float = 0.0
    tokens: int = 0
    chars: int = 0
    tok_s: float = 0.0
    prompt_chars: int = 0
    finish_reason: str = ""


def get_langchain_llm(model: str, temperature: float = 0.2) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI client pointing at Docker Model Runner."""
    return ChatOpenAI(
        model=model,
        base_url=DMR_BASE_URL,
        api_key=SecretStr(DMR_API_KEY),
        temperature=temperature,
    )


def langchain_chat(model: str, system: str, user: str, temperature: float = 0.2) -> str:
    """Send a chat request via LangChain and return the content string."""
    llm = get_langchain_llm(model, temperature)
    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    content = resp.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item if isinstance(item, str) else json.dumps(item) for item in content
        )
    return json.dumps(content)


def _print_progress(msg: str) -> None:
    """Print and flush immediately, thread-safe."""
    with _print_lock:
        print(msg, flush=True)


def _dmr_stream(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    label: str,
    json_mode: bool,
) -> tuple[str, LLMStats]:
    """Stream a chat completion from DMR. Returns (content, stats)."""
    url = f"{DMR_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DMR_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    prompt_chars = len(system) + len(user)
    model_short = model.rsplit("/", 1)[-1].split(":")[0]
    tag = f"[llm][{label}]" if label else "[llm]"
    _print_progress(
        f"{tag} requesting {model_short}  "
        f"prompt={prompt_chars:,} chars  max_tokens={max_tokens}"
    )

    start = time.monotonic()
    r = requests.post(
        url, headers=headers, data=json.dumps(payload),
        timeout=1800, stream=True,
    )
    r.raise_for_status()

    content_parts: list[str] = []
    token_count = 0
    finish_reason = None
    last_report = start

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        delta = chunk.get("choices", [{}])[0].get("delta", {})
        text = delta.get("content", "")
        if text:
            content_parts.append(text)
            token_count += 1

        fr = chunk.get("choices", [{}])[0].get("finish_reason")
        if fr:
            finish_reason = fr

        now = time.monotonic()
        if now - last_report >= 10.0:
            elapsed = now - start
            chars = sum(len(p) for p in content_parts)
            tok_s = token_count / elapsed if elapsed > 0 else 0
            _print_progress(
                f"{tag} {elapsed:5.0f}s  "
                f"tokens={token_count:,}  chars={chars:,}  "
                f"({tok_s:.1f} tok/s)"
            )
            last_report = now

    elapsed = time.monotonic() - start
    content = "".join(content_parts)
    tok_s = token_count / elapsed if elapsed > 0 else 0

    _print_progress(
        f"{tag} done in {elapsed:.1f}s  "
        f"tokens={token_count:,}  chars={len(content):,}  "
        f"({tok_s:.1f} tok/s)  "
        f"finish_reason={finish_reason or 'MISSING'}"
    )

    if finish_reason and finish_reason != "stop":
        _print_progress(
            f"{tag} WARNING: finish_reason='{finish_reason}' — "
            f"response may be truncated"
        )
        _print_progress(f"{tag} content tail: ...{content[-300:]}")

    stats = LLMStats(
        model=model_short,
        label=label,
        elapsed_s=elapsed,
        tokens=token_count,
        chars=len(content),
        tok_s=tok_s,
        prompt_chars=prompt_chars,
        finish_reason=finish_reason or "MISSING",
    )
    return content, stats


def dmr_chat_json(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    label: str = "",
) -> tuple[dict, LLMStats]:
    """Call Docker Model Runner with streaming, return (parsed_dict, stats).

    Uses response_format=json_object. Suitable for structured outputs
    that don't contain complex nested strings (policy, manifest, etc.).
    """
    content, stats = _dmr_stream(
        model, system, user, max_tokens, temperature, label, json_mode=True,
    )
    return extract_json(content), stats


def dmr_chat_raw(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    label: str = "",
) -> tuple[str, LLMStats]:
    """Call Docker Model Runner with streaming, return (raw_text, stats).

    No response_format constraint. Use this when the output contains
    code/file contents that would break JSON escaping.
    """
    content, stats = _dmr_stream(
        model, system, user, max_tokens, temperature, label, json_mode=False,
    )
    return content, stats
