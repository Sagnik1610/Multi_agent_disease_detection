from typing import List, Dict, Union, Sequence, Any, Optional, AsyncGenerator
from autogen_core.models import ChatCompletionClient, ModelInfo, CreateResult, LLMMessage, RequestUsage, ModelCapabilities
from krutrim_cloud import KrutrimCloud

class KrutrimModelClient(ChatCompletionClient):
    def __init__(self, config: Dict):
        self.client = KrutrimCloud()
        self.model = config.get("model")
        self.total_tokens = 0

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool] = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: Optional[Any] = None,
    ) -> CreateResult:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **extra_create_args
        )
        self.total_tokens += response.usage.total_tokens
        return CreateResult(
            messages=[{"role": "assistant", "content": response.choices[0].message.content}],
            usage=RequestUsage(prompt_tokens=response.usage.total_tokens, completion_tokens=0)
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool] = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: Optional[Any] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **extra_create_args
        )
        for chunk in response:
            yield chunk.choices[0].delta.content or ""
        yield CreateResult(
            messages=[{"role": "assistant", "content": ""}],
            usage=RequestUsage(prompt_tokens=0, completion_tokens=0)
        )

    async def close(self) -> None:
        pass

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=self.total_tokens, completion_tokens=0)

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        return 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        }

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "DeepSeek-R1-Llama-8B",
            "structured_output": False,
        }
