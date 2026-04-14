from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class OpenAIClientConfig:
    model_name: str
    base_url: str
    api_key: str
    timeout_seconds: float
    max_tokens: int
    temperature: float


class OpenAICompatibleClient:
    def __init__(self, config: OpenAIClientConfig):
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to query the vLLM OpenAI-compatible server."
            ) from exc

        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            top_p=1.0,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
