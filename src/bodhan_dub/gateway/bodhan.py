"""
Typed API client for Bodhan Indic models:
- Indic-Transcribe (ASR): POST /v1/audio/transcriptions
- Indic-Translate (MT): POST /v1/chat/completions
- Indic-Speak (TTS): POST /v1/audio/speech
- Indic-OCR (Doc OCR): POST /v1/chat/completions (optional)

Encapsulates caching, cost tracking, budget gating, resilience, and policy checking.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Tuple

import httpx

from bodhan_dub.config import Settings
from bodhan_dub.gateway.cache import DiskCache
from bodhan_dub.gateway.cost import BudgetGuard, CostLedger
from bodhan_dub.gateway.resilience import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    execute_with_resilience,
)
from bodhan_dub.logging import PipelineLogger
from bodhan_dub.policy import HostAllowlistGuard


class BodhanClient:
    """Async client managing all interactions with Bodhan AI endpoints."""

    def __init__(
        self,
        settings: Settings,
        cache: DiskCache,
        ledger: CostLedger,
        budget_guard: BudgetGuard,
        logger: PipelineLogger,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.settings = settings
        self.cache = cache
        self.ledger = ledger
        self.budget_guard = budget_guard
        self.logger = logger

        # Concurrency semaphores
        self.sem_asr = asyncio.Semaphore(settings.semaphore_asr)
        self.sem_translate = asyncio.Semaphore(settings.semaphore_translate)
        self.sem_tts = asyncio.Semaphore(settings.semaphore_tts)

        # Circuit breaker registry isolating per (service, language)
        self.cb_registry = CircuitBreakerRegistry(
            failure_threshold=settings.circuit_breaker_threshold,
            recovery_timeout=60.0,
        )

        # Policy guard for allowed hosts
        self.policy_guard = HostAllowlistGuard(
            [
                httpx.URL(settings.bodhan_base_url).host,
                httpx.URL(settings.llm_base_url).host,
            ]
        )

        # Underlying HTTP client with policy hook
        self._http = httpx.AsyncClient(
            base_url=settings.bodhan_base_url,
            timeout=settings.timeout_asr,
            event_hooks={"request": [self.policy_guard.async_request_hook]},
            transport=transport,
        )

    async def close(self) -> None:
        await self._http.aclose()

    @property
    def cb_asr(self) -> CircuitBreaker:
        return self.cb_registry.get_breaker("transcribe", "default")

    @property
    def cb_translate(self) -> CircuitBreaker:
        return self.cb_registry.get_breaker("translate", "default")

    @property
    def cb_tts(self) -> CircuitBreaker:
        return self.cb_registry.get_breaker("speak", "default")

    # -----------------------------------------------------------------------
    # 1. Indic-Transcribe
    # -----------------------------------------------------------------------
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = "en",
        response_format: str = "verbose_json",
        duration_seconds: float = 10.0,
        stage: str = "asr",
        lang: Optional[str] = None,
        segment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes audio using Bodhan Indic-Transcribe.
        Returns parsed response dict containing text and (if verbose_json) segments/words.
        """
        service = "indic-transcribe"
        model = "indic-transcribe"
        params = {"language": language or "auto", "response_format": response_format}

        # Check Cache
        cache_key = self.cache.compute_key(service, model, params, audio_bytes)
        cached = self.cache.get(cache_key)
        if cached is not None:
            data, _ = cached
            await self.ledger.record_transcribe(duration_seconds, from_cache=True)
            self.logger.debug(
                f"Cache hit for transcribe [{cache_key[:8]}]",
                stage=stage,
                lang=lang,
                segment_id=segment_id,
            )
            return data

        # Check Budget
        est_cost = self.ledger.calculate_transcribe_cost(duration_seconds)
        await self.budget_guard.check_budget(est_cost)

        headers = {"Authorization": f"Bearer {self.settings.bodhan_transcribe_key}"}
        form_data = {"model": model, "response_format": response_format}
        if language:
            form_data["language"] = language

        files = {"file": ("chunk.wav", audio_bytes, "audio/wav")}

        async def _call():
            return await self._http.post(
                "/audio/transcriptions",
                headers=headers,
                data=form_data,
                files=files,
                timeout=self.settings.timeout_asr,
            )

        cb = self.cb_registry.get_breaker("transcribe", language or lang or "en")
        resp = await execute_with_resilience(
            _call,
            semaphore=self.sem_asr,
            circuit_breaker=cb,
        )
        resp.raise_for_status()
        data = resp.json()

        # Record actual cost and update cache
        actual_cost = await self.ledger.record_transcribe(duration_seconds, from_cache=False)
        self.cache.put(cache_key, data, actual_cost, extra_meta={"duration": duration_seconds})
        return data

    # -----------------------------------------------------------------------
    # 2. Indic-Translate
    # -----------------------------------------------------------------------
    async def translate(
        self,
        text: str,
        target_language_code: str,
        source_language_code: Optional[str] = "en",
        stage: str = "translate",
        lang: Optional[str] = None,
        segment_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Translates text using Bodhan Indic-Translate.
        Returns (translated_text, usage_dict).
        """
        service = "indic-translate"
        model = "indic-translate"
        params = {
            "source": source_language_code or "auto",
            "target": target_language_code,
        }

        # Check Cache
        cache_key = self.cache.compute_key(service, model, params, text)
        cached = self.cache.get(cache_key)
        if cached is not None:
            data, _ = cached
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            await self.ledger.record_translate(tokens, from_cache=True)
            self.logger.debug(
                f"Cache hit for translate [{cache_key[:8]}]",
                stage=stage,
                lang=lang,
                segment_id=segment_id,
            )
            return data["choices"][0]["message"]["content"], data.get("usage", {})

        # Budget estimate (approx 2 tokens per word)
        est_tokens = max(10, len(text.split()) * 3)
        est_cost = self.ledger.calculate_translate_cost(est_tokens)
        await self.budget_guard.check_budget(est_cost)

        headers = {
            "Authorization": f"Bearer {self.settings.bodhan_translate_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "target_language_code": target_language_code,
        }
        if source_language_code:
            payload["source_language_code"] = source_language_code

        async def _call():
            return await self._http.post(
                "/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.settings.timeout_translate,
            )

        cb = self.cb_registry.get_breaker("translate", target_language_code or "default")
        resp = await execute_with_resilience(
            _call,
            semaphore=self.sem_translate,
            circuit_breaker=cb,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", est_tokens)
        actual_cost = await self.ledger.record_translate(completion_tokens, from_cache=False)

        self.cache.put(cache_key, data, actual_cost)
        translated_content = data["choices"][0]["message"]["content"]
        return translated_content, usage

    # -----------------------------------------------------------------------
    # 3. Indic-Speak
    # -----------------------------------------------------------------------
    async def speak(
        self,
        text: str,
        voice: str,
        language_code: str,
        style: Optional[str] = None,
        stage: str = "tts",
        lang: Optional[str] = None,
        segment_id: Optional[str] = None,
        nonce: Optional[int] = None,
    ) -> bytes:
        """
        Synthesizes text into audio using Bodhan Indic-Speak.
        Returns raw audio bytes (24 kHz mono).
        """
        service = "indic-speak"
        model = "indic-speak"
        inst_dict = {"lang": language_code}
        if style:
            inst_dict["style"] = style
        instructions_str = json.dumps(inst_dict)

        params: Dict[str, Any] = {"voice": voice, "instructions": instructions_str}
        if nonce is not None:
            params["nonce"] = nonce
            self.logger.info(
                f"Deliberate retry_same requested with nonce={nonce} to bypass identical synthesis",
                stage=stage,
                lang=lang,
                segment_id=segment_id,
            )

        # Check Cache
        cache_key = self.cache.compute_key(service, model, params, text)
        cached = self.cache.get(cache_key)
        if cached is not None:
            audio_bytes, _ = cached
            await self.ledger.record_speak(len(text), from_cache=True)
            self.logger.debug(
                f"Cache hit for speak [{cache_key[:8]}]",
                stage=stage,
                lang=lang,
                segment_id=segment_id,
            )
            return audio_bytes

        # Check Budget
        est_cost = self.ledger.calculate_speak_cost(len(text))
        await self.budget_guard.check_budget(est_cost)

        headers = {
            "Authorization": f"Bearer {self.settings.bodhan_speak_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "instructions": instructions_str,
        }

        async def _call():
            return await self._http.post(
                "/audio/speech",
                headers=headers,
                json=payload,
                timeout=self.settings.timeout_tts,
            )

        cb = self.cb_registry.get_breaker("speak", language_code or lang or "default")
        resp = await execute_with_resilience(
            _call,
            semaphore=self.sem_tts,
            circuit_breaker=cb,
        )
        resp.raise_for_status()
        audio_bytes = resp.content

        actual_cost = await self.ledger.record_speak(len(text), from_cache=False)
        self.cache.put(cache_key, audio_bytes, actual_cost, extra_meta={"chars": len(text)})
        return audio_bytes
