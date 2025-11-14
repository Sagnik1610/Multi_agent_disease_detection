from typing import List, Dict, Union, Sequence, Any, Optional, AsyncGenerator
from autogen_core.models import ChatCompletionClient, ModelInfo, CreateResult, LLMMessage, RequestUsage, ModelCapabilities
from krutrim_cloud import KrutrimCloud

class KrutrimModelClient(ChatCompletionClient):
    def __init__(self, config: Dict):
        self.client = KrutrimCloud()
        self.model = config.get("model")
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0

    def _convert_messages_to_dicts(self, messages: Sequence[LLMMessage]) -> List[Dict[str, Any]]:
        """
        Convert LLMMessage objects to dictionaries that the Krutrim API expects.
        
        Args:
            messages: Sequence of LLMMessage objects from autogen
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        converted_messages = []
        
        for msg in messages:
            # Handle different message formats
            if isinstance(msg, dict):
                # Already a dict, just ensure it has required fields
                converted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                # LLMMessage object with attributes
                converted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            elif hasattr(msg, '__dict__'):
                # Object with __dict__, try to extract role and content
                msg_dict = msg.__dict__
                converted_messages.append({
                    "role": msg_dict.get("role", "user"),
                    "content": msg_dict.get("content", "")
                })
            else:
                # Fallback: treat as string content with user role
                converted_messages.append({
                    "role": "user",
                    "content": str(msg)
                })
        
        return converted_messages

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
        # Convert messages to dictionaries
        converted_messages = self._convert_messages_to_dicts(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=converted_messages,
                stream=False,
                **extra_create_args
            )
            
            # Extract content from response
            content = ""
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content or ""
            
            # Extract finish_reason
            finish_reason = "stop"  # default
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'finish_reason'):
                    # Map the finish reason to autogen's expected values
                    fr = choice.finish_reason
                    if fr in ["stop", "length", "function_calls", "content_filter"]:
                        finish_reason = fr
                    else:
                        finish_reason = "stop"  # default for unknown reasons
            
            # Track usage
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                self._total_prompt_tokens += prompt_tokens
                self._total_completion_tokens += completion_tokens
                self._total_tokens += getattr(response.usage, 'total_tokens', 0)
            
            return CreateResult(
                finish_reason=finish_reason,
                content=content,
                usage=RequestUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                ),
                cached=False
            )
        except Exception as e:
            raise RuntimeError(f"Krutrim API call failed: {str(e)}") from e

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
        # Convert messages to dictionaries
        converted_messages = self._convert_messages_to_dicts(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=converted_messages,
                stream=True,
                **extra_create_args
            )
            
            full_content = ""
            for chunk in response:
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, 'content', None) or ""
                    if content:
                        full_content += content
                        yield content
            
            yield CreateResult(
                finish_reason="stop",
                content=full_content,
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False
            )
        except Exception as e:
            raise RuntimeError(f"Krutrim streaming API call failed: {str(e)}") from e

    async def close(self) -> None:
        pass

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_prompt_tokens,
            completion_tokens=self._total_completion_tokens
        )

    def total_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_prompt_tokens,
            completion_tokens=self._total_completion_tokens
        )

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        # Rough estimation: 4 characters per token
        total_chars = sum(len(str(msg)) for msg in messages)
        return total_chars // 4

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        # Assuming a context window of 8192 tokens for DeepSeek-R1-Llama-8B
        max_tokens = 8192
        used_tokens = self.count_tokens(messages, tools=tools)
        return max(0, max_tokens - used_tokens)

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
