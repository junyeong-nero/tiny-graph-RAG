"""OpenAI API client wrapper."""

import json
from openai import OpenAI


class OpenAIClient:
    """Wrapper for OpenAI API calls."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use for completions
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
