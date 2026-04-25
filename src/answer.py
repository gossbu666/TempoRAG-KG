"""Answer generation: (question, retrieved chunks) → answer string.

Provider-agnostic. `answer_question` takes anything conforming to the
`LLMClient` protocol (`.generate(prompt, temperature) -> str`). Concrete
adapters in this module:

- `GroqClient`  — `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`,
                  `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
                  via Groq's OpenAI-compatible endpoint
                  (`https://api.groq.com/openai/v1`).
- (Existing `OpenAIClient` in `src.kg_extract` handles OpenAI models.)

Cache semantics mirror `kg_extract`: **raw response in, parsed answer out.**
If we tighten the JSON parser later, cached responses re-parse for free.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

from src.cache import Cache


class LLMClient(Protocol):
    def generate(self, prompt: str, temperature: float = 0.0) -> str: ...


# -----------------------------------------------------------------------------
# Prompt rendering
# -----------------------------------------------------------------------------

def render_answer_prompt(
    template: str,
    question: str,
    retrieved_chunks: list[dict],
    *,
    text_field: str = "text",
) -> str:
    """Substitute {{QUESTION}} + {{CONTEXT}} placeholders.

    Each retrieved chunk is rendered with a human-readable header
    (chunk_id | ticker FY | item) followed by its text. Headers help the
    model cite, and also help us debug which chunks the answer came from.
    """
    if "{{QUESTION}}" not in template or "{{CONTEXT}}" not in template:
        raise ValueError(
            "answer template must contain both {{QUESTION}} and {{CONTEXT}}"
        )
    blocks: list[str] = []
    for i, c in enumerate(retrieved_chunks, 1):
        header = (
            f"[Chunk {i} | id={c.get('chunk_id','?')} | "
            f"{c.get('ticker','?')} FY{c.get('fy','?')} Item {c.get('item','?')}]"
        )
        blocks.append(f"{header}\n{c.get(text_field, '')}")
    context = "\n\n".join(blocks)
    return (
        template
        .replace("{{QUESTION}}", question)
        .replace("{{CONTEXT}}", context)
    )


def load_answer_template(
    path: Path | str = "prompts/answer_v1.txt",
) -> str:
    return Path(path).read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE
)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    return m.group(1).strip() if m else stripped


def parse_answer(raw: str) -> tuple[str, str | None]:
    """Extract `answer` string from the model's JSON reply.

    Returns (answer, error). On parse failure we fall back to the stripped
    raw text (with an error string set) rather than raising — a badly
    formatted answer is a data point, not a crash. Downstream eval can
    still score it.
    """
    body = _strip_fence(raw)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return raw.strip(), f"JSONDecodeError: {e.msg} at pos {e.pos}"
    if not isinstance(obj, dict):
        return raw.strip(), (
            f"top-level JSON must be an object, got {type(obj).__name__}"
        )
    ans = obj.get("answer")
    if not isinstance(ans, str):
        return raw.strip(), f"'answer' missing or not a string, got {type(ans).__name__}"
    return ans.strip(), None


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def answer_question(
    question: str,
    retrieved_chunks: list[dict],
    client: LLMClient,
    cache: Cache,
    *,
    template: str,
    model: str,
    temperature: float = 0.0,
) -> dict:
    """Call the LLM with (question, retrieved context); return answer + meta.

    Returns:
      {"answer": str, "raw_response": str, "cache_hit": bool,
       "parse_error": str | None, "n_chunks": int}

    Cache semantics (inherited from kg_extract):
      key = sha256(model | rendered_prompt | {"temperature": t}).
      Prompt rendering embeds the chunk texts, so different retrieved sets
      produce different keys automatically.
    """
    rendered = render_answer_prompt(template, question, retrieved_chunks)
    # Include max_tokens in cache key so raising the cap for reasoning models
    # (gpt-oss family, where 400 is exhausted by CoT tokens and content ends
    # up empty) auto-invalidates stale empty responses. Models without an
    # exposed cap fall back to a sentinel so the key stays stable.
    client_max_tokens = getattr(client, "_max_tokens", None)
    params: dict = {"temperature": temperature}
    if client_max_tokens is not None:
        params["max_tokens"] = client_max_tokens
    key = cache.key_for(model, rendered, params)

    cached = cache.get(key)
    if cached is not None:
        raw = cached["response"]
        ans, err = parse_answer(raw)
        return {
            "answer": ans, "raw_response": raw, "cache_hit": True,
            "parse_error": err, "n_chunks": len(retrieved_chunks),
        }

    raw = client.generate(rendered, temperature=temperature)
    cache.put(key, {"response": raw, "model": model})
    ans, err = parse_answer(raw)
    return {
        "answer": ans, "raw_response": raw, "cache_hit": False,
        "parse_error": err, "n_chunks": len(retrieved_chunks),
    }


# -----------------------------------------------------------------------------
# Groq adapter (OpenAI-compatible API)
# -----------------------------------------------------------------------------
# Groq serves Llama and GPT-OSS models through an OpenAI-compatible endpoint,
# so we can reuse the `openai` SDK by passing `base_url`. The interface is
# intentionally identical to `OpenAIClient` in `src.kg_extract`.
# -----------------------------------------------------------------------------


class GroqClient:
    """Thin adapter around Groq's OpenAI-compatible chat completion endpoint.

    Currently unused (Groq Developer-tier upgrades paused 2026-04; the
    free-tier TPD budget can't cover our full sweep). Kept for the case
    when Groq re-opens paid tier or we want to run a subset on free tier.

    `max_tokens=400` caps runaway outputs — answer prompts ask for 1-2
    sentences, and verbose models (GPT-OSS especially) otherwise tack on
    explanatory text that inflates cost without helping F1.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        max_tokens: int = 400,
    ) -> None:
        from openai import OpenAI  # deferred — keeps test imports cheap
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GROQ_API_KEY not set — refuse to run without a key"
            )
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=resolved_key, base_url=base_url)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class OpenRouterClient:
    """Thin adapter around OpenRouter's OpenAI-compatible endpoint.

    Routes the 4 open-weights models (Llama 3.1 8B / 3.3 70B, GPT-OSS 20B
    /120B) here because Groq Developer-tier upgrades are paused. Pricing
    at OpenRouter is competitive (cents per full sweep) and the API is
    drop-in OpenAI-compatible, so this class is structurally identical to
    `GroqClient` — only the base URL + env var change.

    A common OpenRouter pitfall: some models error on
    `response_format={"type":"json_object"}` because the underlying host
    doesn't support structured outputs. We send it anyway and rely on
    the prompt to produce JSON; if the model wraps prose around the JSON,
    `parse_answer` strips a code fence and extracts the answer.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 400,
    ) -> None:
        from openai import OpenAI  # deferred
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set — refuse to run without a key"
            )
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=resolved_key, base_url=base_url)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""


# -----------------------------------------------------------------------------
# Factory — build client + register model metadata for the eval runner
# -----------------------------------------------------------------------------

# Maps friendly config names to (provider, model_id). The eval runner takes a
# config-name list and instantiates the right client per row. Keeps the matrix
# declaration in one place so progress reports and run scripts stay in sync.
#
# Provider routing (2026-04-20):
#   - open-weights Llama + GPT-OSS → OpenRouter (Groq dev-tier paused)
#   - OpenAI closed models         → OpenAI direct
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # ─ small tier ──────────────────────────────────────────────────────────
    "llama-8b":     ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
    "gpt-oss-20b":  ("openrouter", "openai/gpt-oss-20b"),
    "gpt-4.1-nano": ("openai",     "gpt-4.1-nano"),
    # ─ large tier ──────────────────────────────────────────────────────────
    "llama-70b":    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    "gpt-oss-120b": ("openrouter", "openai/gpt-oss-120b"),
    "gpt-4o-mini":  ("openai",     "gpt-4o-mini"),
    # ─ flagship ceiling ────────────────────────────────────────────────────
    "gpt-4o":       ("openai",     "gpt-4o"),
}

# Per-model max_tokens override. GPT-OSS is a reasoning-model family — its
# chain-of-thought counts against max_tokens and a 400-token cap empties
# `message.content` ~34% of the time. 2000 leaves ~1500 for reasoning + ~500
# for the short JSON answer; verified sufficient on 5-chunk smoke tests.
# Non-reasoning models keep the 400 default (cheap + enforces our "1-2
# sentence" prompt discipline).
MODEL_MAX_TOKENS: dict[str, int] = {
    "gpt-oss-20b":  2000,
    "gpt-oss-120b": 2000,
}


def build_client(config_name: str) -> LLMClient:
    """Instantiate the right adapter for a registered config name."""
    if config_name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model config {config_name!r}; "
            f"valid keys: {sorted(MODEL_REGISTRY)}"
        )
    provider, model_id = MODEL_REGISTRY[config_name]
    max_tokens = MODEL_MAX_TOKENS.get(config_name, 400)
    if provider == "groq":
        return GroqClient(model_name=model_id, max_tokens=max_tokens)
    if provider == "openrouter":
        return OpenRouterClient(model_name=model_id, max_tokens=max_tokens)
    if provider == "openai":
        # Reuse existing OpenAIClient from kg_extract — same interface.
        from src.kg_extract import OpenAIClient
        return OpenAIClient(model_name=model_id)
    raise ValueError(f"unknown provider {provider!r} for {config_name!r}")
