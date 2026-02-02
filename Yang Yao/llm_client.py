from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Type, TypeVar, Union

from pydantic import BaseModel

from openai import OpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

from settings import settings

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """Unified wrapper to make agent logging/debugging easier later."""
    text: str
    model: str
    usage: Optional[dict[str, Any]] = None
    raw: Optional[Any] = None


class LLMClient:
    """
    An LLM client wrapper:
    - Supports text(): standard chat/text completion output
    - Supports json(): force JSON output + optional Pydantic validation
    - Built-in retries (rate limits / timeouts / transient server errors)
    - For future agent/tool calling, you can add tools/tool_choice directly in chat()
    """


    def __init__(
        self,
        model: Optional[str] = None,
        model_strong: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_s: Optional[int] = None,
        max_retries: int = 5,
    ) -> None:
        self.model = model or settings.OPENAI_MODEL
        self.model_strong = model_strong or settings.OPENAI_MODEL_STRONG
        self.temperature = float(temperature if temperature is not None else settings.OPENAI_TEMPERATURE)
        self.timeout_s = int(timeout_s if timeout_s is not None else settings.OPENAI_TIMEOUT_S)
        self.max_retries = int(max_retries)

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())

    # -------------------------
    # Public APIs
    # -------------------------

    def text(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        resp = self.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=model or self.model,
            temperature=self.temperature if temperature is None else float(temperature),
        )
        return resp.text

    def json(
        self,
        prompt: str,
        *,
        schema: Optional[Type[T]] = None,
        system: str = "You are a helpful assistant. Return ONLY valid JSON.",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        repair_attempts: int = 1,
    ) -> Union[dict[str, Any], T]:
        """
        Force the model to output JSON only.
        - schema=None -> returns a dict
        - schema=a Pydantic model -> returns a model instance (will attempt repair if validation fails)
        """

        schema_hint = ""
        if schema is not None:
            schema_json = schema.model_json_schema()
            schema_hint = (
                "\n\nYou MUST output a JSON object that matches this JSON Schema:\n"
                + json.dumps(schema_json, ensure_ascii=False)
                + "\nDo not add extra keys unless schema allows it."
            )

        def _call(with_extra: str = "") -> str:
            resp = self.chat(
                messages=[
                    {"role": "system", "content": system + schema_hint},
                    {"role": "user", "content": prompt + with_extra},
                ],
                model=model or self.model,
                temperature=self.temperature if temperature is None else float(temperature),
                response_format={"type": "json_object"},
            )
            return resp.text

        raw = _call()

        # 1) First try to parse directly as a dict
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            # Ask the model to repair once
            if repair_attempts <= 0:
                raise RuntimeError(f"Model did not return valid JSON. Raw:\n{raw}") from e
            raw = _call(
                with_extra="\n\nYour previous output was not valid JSON. Please output ONLY valid JSON."
            )
            obj = json.loads(raw)

        # 2) If a schema is provided, validate with Pydantic; if it fails, attempt repair
        if schema is None:
            return obj

        try:
            return schema.model_validate(obj)
        except Exception as e:
            if repair_attempts <= 0:
                raise RuntimeError(
                    "JSON returned but does not match schema.\n"
                    f"Raw JSON:\n{json.dumps(obj, ensure_ascii=False, indent=2)}"
                ) from e

            raw2 = _call(
                with_extra=(
                    "\n\nThe JSON you returned does NOT match the schema. "
                    "Fix it and output ONLY JSON that matches the schema."
                )
            )
            obj2 = json.loads(raw2)
            return schema.model_validate(obj2)

    def chat(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        model: str,
        temperature: float,
        response_format: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Unified low-level call wrapper.

        For tool calling (used by agents), pass `tools` and `tool_choice` directly.
        """

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "timeout": self.timeout_s,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        result = self._with_retries(lambda: self.client.chat.completions.create(**kwargs))

        # Compatibility: handle OpenAI SDK response structure
        text = (result.choices[0].message.content or "").strip()
        usage = None
        try:
            usage = result.usage.model_dump() if result.usage else None
        except Exception:
            usage = None

        return LLMResponse(text=text, model=model, usage=usage, raw=result)

    # -------------------------
    # Internal: retries
    # -------------------------

    def _with_retries(self, fn):
        last_err: Optional[Exception] = None
        for i in range(self.max_retries + 1):
            try:
                return fn()
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
                last_err = e
                # Exponential backoff + jitter
                sleep_s = min(2 ** i, 20) + (0.1 * i)
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI request failed after retries: {last_err}") from last_err
