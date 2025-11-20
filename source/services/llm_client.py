# backend/services/llm_client.py (async version)

from typing import List
import os
import logging
import json
import httpx
from settings import settings

logger = logging.getLogger(__name__)

# Try import OpenAI async SDK
try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    AsyncOpenAI = None
    _OPENAI_AVAILABLE = False


# ENV / settings
LLM_API_KEY = settings.LLM_API_KEY
LLM_BASE_URL = settings.LLM_BASE_URL

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss-20b")
HTTP_TIMEOUT = float(os.getenv("LLM_HTTP_TIMEOUT", "30"))


def _build_prompt(question: str, context_texts: List[str]) -> str:
    """Format prompt untuk LLM."""
    contexts = context_texts[:3] or ["No context available."]
    context_block = "\n\n".join(contexts)

    return (
        "You are a document assistant. Use the context below to answer the user's question.\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question}"
    )


# ============================================================================
# ASYNC VERSION
# ============================================================================
async def ask_llm(question: str, context_texts: List[str]) -> str:
    """
    Async LLM caller:
    - Build prompt
    - Use async OpenAI SDK if installed
    - Fallback to async HTTP POST
    """
    prompt = _build_prompt(question, context_texts)

    # ----------------------------------------------------------------------
    # 1) TRY OPENAI Async SDK
    # ----------------------------------------------------------------------
    if _OPENAI_AVAILABLE and AsyncOpenAI is not None:
        try:
            client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

            resp = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.0,
            )

            answer = resp.choices[0].message.content.strip()
            return answer

        except Exception as e:
            logger.exception(f"AsyncOpenAI failed: {e}")

    # ----------------------------------------------------------------------
    # 2) FALLBACK ASYNC HTTP CALL
    # ----------------------------------------------------------------------
    try:
        url = LLM_BASE_URL.rstrip("/") + "/chat/completions"

        headers = {"Content-Type": "application/json"}
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                logger.error(f"HTTP LLM failed: {resp.text}")
                raise RuntimeError(f"LLM HTTP error {resp.status_code}")

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip()

    except Exception as e:
        logger.exception(f"Async HTTP fallback failed: {e}")
        raise RuntimeError(f"Failed to get LLM answer: {e}")
