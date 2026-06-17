"""
LLM Client — Wrapper around vLLM's OpenAI-compatible API.
Handles retries, JSON mode, token tracking, and error logging.
"""
import json
import time
from openai import OpenAI
from src.config import (
    VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_NAME,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_RETRIES,
)


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (~4 chars per token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class LLMClient:
    """Wrapper for vLLM OpenAI-compatible API with retry logic and token tracking."""

    def __init__(self, base_url=None, model_name=None):
        self.client = OpenAI(
            base_url=base_url or VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
        )
        self.model_name = model_name or VLLM_MODEL_NAME
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_log = []  # Per-call timing and token log
        self._start_time = time.time()

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

        call_start = time.time()

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                call_duration = time.time() - call_start
                self.call_count += 1

                response_text = response.choices[0].message.content

                # Track tokens — use API response if available, else estimate
                if response.usage and response.usage.total_tokens:
                    input_tok = response.usage.prompt_tokens or 0
                    output_tok = response.usage.completion_tokens or 0
                    total_tok = response.usage.total_tokens
                else:
                    # vLLM often returns None for usage — estimate manually
                    input_tok = _estimate_tokens(system) + _estimate_tokens(user)
                    output_tok = _estimate_tokens(response_text)
                    total_tok = input_tok + output_tok

                self.total_input_tokens += input_tok
                self.total_output_tokens += output_tok
                self.total_tokens += total_tok

                # Log this call
                self.call_log.append({
                    "call_id": self.call_count,
                    "input_tokens": input_tok,
                    "output_tokens": output_tok,
                    "total_tokens": total_tok,
                    "duration_sec": round(call_duration, 2),
                    "timestamp": time.time(),
                })

                return response_text

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
        elapsed = time.time() - self._start_time
        return {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_tokens_per_call": round(self.total_tokens / max(self.call_count, 1)),
            "avg_latency_sec": round(
                sum(c["duration_sec"] for c in self.call_log) / max(self.call_count, 1), 2
            ),
            "total_llm_time_sec": round(sum(c["duration_sec"] for c in self.call_log), 1),
            "wall_clock_sec": round(elapsed, 1),
        }

    def get_call_log(self) -> list:
        """Return the per-call log for detailed metrics."""
        return self.call_log
