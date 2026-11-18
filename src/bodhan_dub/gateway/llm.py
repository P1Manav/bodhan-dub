"""
OpenAI-compatible LLM gateway with strict English-only enforcement,
JSON-in-text schema validation, 1 repair retry, and deterministic fallback.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from bodhan_dub.config import Settings
from bodhan_dub.gateway.cache import DiskCache
from bodhan_dub.gateway.cost import CostLedger
from bodhan_dub.logging import PipelineLogger
from bodhan_dub.policy import PolicyViolation, contains_indic_or_arabic, validate_llm_text

T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extracts first valid JSON object found in text (including within ```json fences)."""
    # 1. Match code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    # 2. Match outermost curly braces
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass

    return None


class LLMClient:
    """Manages English-only LLM queries with JSON validation and graceful fallback."""

    def __init__(
        self,
        settings: Settings,
        cache: DiskCache,
        ledger: CostLedger,
        logger: PipelineLogger,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.settings = settings
        self.cache = cache
        self.ledger = ledger
        self.logger = logger
        self.sem_llm = asyncio.Semaphore(settings.semaphore_llm)

        self._http = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            timeout=settings.timeout_llm,
            transport=transport,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def query_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        fallback_value: T,
        stage: str = "llm",
        lang: Optional[str] = None,
        segment_id: Optional[str] = None,
    ) -> T:
        """
        Sends query to LLM, validating that:
        1. Prompts contain NO Indic or Arabic characters
        2. Response parses into JSON matching Pydantic schema
        3. Response contains NO Indic or Arabic characters
        Retries once with error repair on JSON failure, then returns fallback_value.
        """
        # Validate inputs are English-only
        try:
            validate_llm_text(system_prompt, context="LLM system prompt")
            validate_llm_text(user_prompt, context="LLM user prompt")
        except PolicyViolation as e:
            self.logger.error(
                f"Policy Guard rejected LLM call: {e}",
                stage=stage,
                lang=lang,
                segment_id=segment_id,
            )
            return fallback_value

        # Check Cache
        cache_key = self.cache.compute_key(
            service="llm",
            model=self.settings.llm_model,
            params={"schema": schema.__name__},
            payload=f"{system_prompt}\n---\n{user_prompt}",
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            data, _ = cached
            try:
                parsed = schema.model_validate(data)
                self.logger.debug(
                    "Cache hit for LLM query", stage=stage, lang=lang, segment_id=segment_id
                )
                return parsed
            except Exception:
                pass

        if not self.settings.llm_api_key:
            self.logger.warning(
                "LLM_API_KEY not set; using deterministic fallback",
                stage=stage,
                lang=lang,
                segment_id=segment_id,
            )
            return fallback_value

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        async def _call_api(msgs: list) -> Optional[str]:
            headers = {
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.settings.llm_model,
                "messages": msgs,
                "temperature": 0.0,
            }
            async with self.sem_llm:
                resp = await self._http.post("/chat/completions", headers=headers, json=body)
                if resp.status_code != 200:
                    self.logger.warning(
                        f"LLM API error {resp.status_code}: {resp.text[:200]}",
                        stage=stage,
                        lang=lang,
                        segment_id=segment_id,
                    )
                    return None
                data = resp.json()
                usage = data.get("usage", {})
                await self.ledger.record_llm_tokens(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )
                return data["choices"][0]["message"]["content"]

        try:
            raw_output = await _call_api(messages)
            if not raw_output:
                return fallback_value

            # Policy Guard check on output: MUST NOT contain Indic/Arabic
            if contains_indic_or_arabic(raw_output):
                self.logger.error(
                    "LLM output violated Policy Guard (contained Indic/Arabic text). Discarding.",
                    stage=stage,
                    lang=lang,
                    segment_id=segment_id,
                )
                return fallback_value

            # Attempt JSON parse
            parsed_json = extract_json_from_text(raw_output)
            if parsed_json:
                try:
                    obj = schema.model_validate(parsed_json)
                    self.cache.put(cache_key, parsed_json, cost=0.0)
                    return obj
                except ValidationError:
                    pass

            # One repair retry
            self.logger.warning(
                "LLM response failed schema validation. Attempting 1 repair retry...",
                stage=stage,
                lang=lang,
                segment_id=segment_id,
            )
            repair_messages = list(messages)
            repair_messages.append({"role": "assistant", "content": raw_output or ""})
            repair_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON conforming to the schema. "
                        f"Return ONLY valid raw JSON conforming to schema: {schema.model_json_schema()}."
                    ),
                }
            )
            repaired_output = await _call_api(repair_messages)
            if repaired_output and not contains_indic_or_arabic(repaired_output):
                repaired_json = extract_json_from_text(repaired_output)
                if repaired_json:
                    obj = schema.model_validate(repaired_json)
                    self.cache.put(cache_key, repaired_json, cost=0.0)
                    return obj

        except Exception as e:
            self.logger.warning(
                f"LLM call encountered exception: {e}; using fallback",
                stage=stage,
                lang=lang,
                segment_id=segment_id,
            )

        return fallback_value

    async def complete_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        stage: str = "llm",
        lang: Optional[str] = None,
        segment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method that queries LLM for a JSON object matching an arbitrary dict schema.
        """
        try:
            validate_llm_text(prompt, context="LLM prompt")
        except PolicyViolation as e:
            self.logger.error(
                f"Policy Guard rejected LLM call: {e}",
                stage=stage,
                lang=lang,
                segment_id=segment_id,
            )
            return {}

        schema_desc = json.dumps(schema) if schema else "{}"
        system_prompt = (
            "You are a helpful linguistic assistant. Return ONLY valid JSON adhering to the specified schema. "
            f"Expected schema: {schema_desc}. Do not include markdown code fences or any non-English text."
        )

        class GenericDictResponse(BaseModel):
            class Config:
                extra = "allow"

        res = await self.query_json(
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=GenericDictResponse,
            fallback_value=GenericDictResponse(),
            stage=stage,
            lang=lang,
            segment_id=segment_id,
        )
        return res.model_dump()
