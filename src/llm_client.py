"""
LLM Client — Wrapper around vLLM's OpenAI-compatible API.
Handles retries, JSON mode, and error logging.
"""
import json
import time
from openai import OpenAI
from src.config import (
    VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_NAME,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_RETRIES,
)


class LLMClient:
    """Wrapper for vLLM OpenAI-compatible API with retry logic."""

    def __init__(self, base_url=None, model_name=None):
        self.client = OpenAI(
            base_url=base_url or VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
        )
        self.model_name = model_name or VLLM_MODEL_NAME
        self.call_count = 0
        self.total_tokens = 0

    def call(
        self,
        system: str,
        user: str,
        json_mode: bool = True,
        temperature: float = None,
        max_tokens: int = None,
        retries: int = None,
    ) -> str:
        """
        Make an LLM call with retry logic.

        Args:
            system: System prompt
            user: User prompt
            json_mode: If True, request JSON output format
            temperature: Override default temperature
            max_tokens: Override default max tokens
            retries: Override default retry count

        Returns:
            LLM response text (string)
        """
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        max_tokens = max_tokens or LLM_MAX_TOKENS
        retries = retries or LLM_MAX_RETRIES

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                self.call_count += 1
                if response.usage:
                    self.total_tokens += response.usage.total_tokens

                return response.choices[0].message.content

            except Exception as e:
                print(f"  [LLM] Attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"  [LLM] Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM call failed after {retries} attempts: {e}"
                    )

    def call_json(self, system: str, user: str, **kwargs) -> dict:
        """Make an LLM call and parse JSON response."""
        response = self.call(system=system, user=user, json_mode=True, **kwargs)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                raise ValueError(f"Failed to parse LLM response as JSON: {response[:200]}")

    def health_check(self) -> bool:
        """Check if the vLLM server is running and responsive."""
        try:
            response = self.call(
                system="You are a helpful assistant.",
                user="Respond with exactly: {\"status\": \"ok\"}",
                json_mode=True,
                max_tokens=20,
            )
            result = json.loads(response)
            return result.get("status") == "ok"
        except Exception as e:
            print(f"[LLM] Health check failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
        }
