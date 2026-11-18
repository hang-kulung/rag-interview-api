import json
from collections.abc import Sequence
from typing import Protocol, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.core.exceptions import LLMError
from app.schemas.chat import ChatMessage, Role

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    async def generate(self, *, system: str, messages: Sequence[ChatMessage]) -> str: ...

    async def generate_structured(
        self, *, system: str, messages: Sequence[ChatMessage], schema: type[T]
    ) -> T: ...


class GeminiLLMClient:
    def __init__(self, client: genai.Client, *, model: str, temperature: float = 0.2) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def generate(self, *, system: str, messages: Sequence[ChatMessage]) -> str:
        return await self._call(system=system, messages=messages, json_mode=False)

    async def generate_structured(
        self, *, system: str, messages: Sequence[ChatMessage], schema: type[T]
    ) -> T:
        schema_hint = json.dumps(schema.model_json_schema())
        text = await self._call(
            system=f"{system}\n\nRespond with a single JSON object matching this JSON schema:\n"
            f"{schema_hint}",
            messages=messages,
            json_mode=True,
        )
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(f"LLM returned JSON that does not match the schema: {exc}") from exc

    async def _call(self, *, system: str, messages: Sequence[ChatMessage], json_mode: bool) -> str:
        trimmed = list(messages)
        while trimmed and trimmed[0].role is Role.ASSISTANT:
            trimmed.pop(0)
        contents = [
            types.Content(
                role="user" if message.role is Role.USER else "model",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in trimmed
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self._temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        except genai_errors.APIError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if not response.text:
            raise LLMError("LLM returned an empty response.")
        return response.text
