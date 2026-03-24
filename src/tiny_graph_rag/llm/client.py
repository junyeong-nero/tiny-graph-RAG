"""OpenAI API client wrapper."""

import json
from typing import Optional

from openai import AsyncOpenAI, OpenAI


class OpenAIClient:
    """Wrapper for OpenAI API calls."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        default_temperature: float = 0.0,
    ):
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use for completions
            base_url: Optional base URL for OpenAI-compatible APIs
            max_tokens: Maximum tokens for responses
            default_temperature: Default sampling temperature
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.default_temperature = default_temperature

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat completion request.

        Args:
            system_prompt: System message for the conversation
            user_prompt: User message
            temperature: Sampling temperature

        Returns:
            The assistant's response text
        """
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        if "gpt-5" in self.model:
            kwargs.pop("temperature")

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict:
        """Send a chat completion request expecting JSON response.

        Args:
            system_prompt: System message for the conversation
            user_prompt: User message
            temperature: Sampling temperature

        Returns:
            Parsed JSON response as dictionary
        """
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        if "gpt-5" in self.model:
            kwargs.pop("temperature")

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    async def async_chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict:
        """Send an async chat completion request expecting JSON response.

        Args:
            system_prompt: System message for the conversation
            user_prompt: User message
            temperature: Sampling temperature

        Returns:
            Parsed JSON response as dictionary
        """
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        if "gpt-5" in self.model:
            kwargs.pop("temperature")

        response = await self.async_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
