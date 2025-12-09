"""HTTP REST API mock model provider.

This module provides a model that wraps a standard HTTP REST API to mock
a standard LLM. It sends chat messages to a configurable HTTP endpoint
and expects responses in a standard format.
"""

import os
from logging import getLogger
from typing import Any

import httpx
from typing_extensions import override

from inspect_ai.tool import ToolChoice, ToolInfo

from .._chat_message import ChatMessage, ChatMessageAssistant
from .._generate_config import GenerateConfig
from .._model import ModelAPI
from .._model_call import ModelCall
from .._model_output import (
    ChatCompletionChoice,
    ModelOutput,
    ModelUsage,
    StopReason,
    as_stop_reason,
)

logger = getLogger(__name__)


class MockRestAPI(ModelAPI):
    """A model implementation that wraps a standard HTTP REST API.

    This model sends chat messages to a configurable HTTP endpoint and
    expects responses in a standard format. It can be used to mock an LLM
    by pointing it at any HTTP server that implements the expected API.

    The expected request format is:
    ```json
    {
        "model": "model-name",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "tools": [...],
        "tool_choice": "auto" | "none" | {"type": "function", "function": {"name": "..."}},
        "temperature": 0.7,
        "max_tokens": 1024,
        ...
    }
    ```

    The expected response format is:
    ```json
    {
        "id": "response-id",
        "model": "model-name",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "response text"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    ```
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        timeout: float = 60.0,
        **model_args: Any,
    ) -> None:
        """Create a MockRestAPI model.

        Args:
            model_name: Name of the model to use in requests.
            base_url: Base URL for the REST API endpoint. Can also be set via
                MOCKREST_BASE_URL environment variable.
            api_key: API key for authentication. Can also be set via
                MOCKREST_API_KEY environment variable.
            config: Generation configuration.
            timeout: Request timeout in seconds.
            **model_args: Additional arguments to pass to the API.
        """
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=["MOCKREST_API_KEY"],
            config=config,
        )

        # Resolve base_url from environment if not provided
        if not self.base_url:
            self.base_url = os.environ.get("MOCKREST_BASE_URL", None)

        if not self.base_url:
            raise ValueError(
                "MockRestAPI requires a base_url. Set it via the base_url parameter "
                "or the MOCKREST_BASE_URL environment variable."
            )

        # Resolve api_key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("MOCKREST_API_KEY", None)

        self.timeout = timeout
        self.model_args = model_args

        # Create HTTP client
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

    @override
    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    @override
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        """Generate a response from the REST API.

        Args:
            input: List of chat messages.
            tools: List of available tools.
            tool_choice: Tool choice configuration.
            config: Generation configuration.

        Returns:
            ModelOutput with the generated response.
        """
        # Build request payload
        request_payload = self._build_request(input, tools, tool_choice, config)

        # Track request/response for ModelCall
        response_data: dict[str, Any] = {}

        def model_call() -> ModelCall:
            return ModelCall.create(
                request=request_payload,
                response=response_data,
            )

        try:
            # Make the HTTP request
            headers = self._build_headers()
            response = await self._client.post(
                "/v1/chat/completions",
                json=request_payload,
                headers=headers,
            )
            response.raise_for_status()
            response_data = response.json()

            # Parse the response
            output = self._parse_response(response_data)
            return output, model_call()

        except httpx.HTTPStatusError as ex:
            error_msg = f"HTTP error {ex.response.status_code}: {ex.response.text}"
            logger.error(error_msg)
            return (
                ModelOutput.from_content(
                    model=self.model_name,
                    content=error_msg,
                    stop_reason="unknown",
                    error=error_msg,
                ),
                model_call(),
            )
        except httpx.RequestError as ex:
            error_msg = f"Request error: {str(ex)}"
            logger.error(error_msg)
            return (
                ModelOutput.from_content(
                    model=self.model_name,
                    content=error_msg,
                    stop_reason="unknown",
                    error=error_msg,
                ),
                model_call(),
            )
        except Exception as ex:
            error_msg = f"Unexpected error: {str(ex)}"
            logger.error(error_msg)
            return (
                ModelOutput.from_content(
                    model=self.model_name,
                    content=error_msg,
                    stop_reason="unknown",
                    error=error_msg,
                ),
                model_call(),
            )

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the request."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_request(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> dict[str, Any]:
        """Build the request payload for the REST API."""
        # Convert messages to standard format
        messages = []
        for msg in input:
            message_dict: dict[str, Any] = {
                "role": msg.role,
                "content": msg.text if hasattr(msg, "text") else str(msg.content),
            }
            messages.append(message_dict)

        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        # Add tools if provided
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        # Add tool_choice if provided
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                request["tool_choice"] = tool_choice
            elif isinstance(tool_choice, dict):
                request["tool_choice"] = tool_choice

        # Add generation config parameters
        if config.temperature is not None:
            request["temperature"] = config.temperature
        if config.max_tokens is not None:
            request["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            request["top_p"] = config.top_p
        if config.top_k is not None:
            request["top_k"] = config.top_k
        if config.stop_seqs is not None:
            request["stop"] = config.stop_seqs
        if config.seed is not None:
            request["seed"] = config.seed

        # Add any additional model args
        request.update(self.model_args)

        return request

    def _parse_response(self, response: dict[str, Any]) -> ModelOutput:
        """Parse the REST API response into a ModelOutput."""
        choices = []
        for choice_data in response.get("choices", []):
            message_data = choice_data.get("message", {})
            content = message_data.get("content", "")
            finish_reason = choice_data.get("finish_reason", "stop")

            # Create assistant message
            assistant_message = ChatMessageAssistant(
                content=content,
                model=response.get("model", self.model_name),
                source="generate",
            )

            # Parse tool calls if present
            tool_calls_data = message_data.get("tool_calls", [])
            if tool_calls_data:
                from inspect_ai.tool._tool_call import ToolCall

                tool_calls = []
                for tc in tool_calls_data:
                    function_data = tc.get("function", {})
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            function=function_data.get("name", ""),
                            arguments=function_data.get("arguments", {}),
                            type=tc.get("type", "function"),
                        )
                    )
                assistant_message = ChatMessageAssistant(
                    content=content,
                    model=response.get("model", self.model_name),
                    source="generate",
                    tool_calls=tool_calls,
                )

            choices.append(
                ChatCompletionChoice(
                    message=assistant_message,
                    stop_reason=self._map_finish_reason(finish_reason),
                )
            )

        # Parse usage if present
        usage = None
        usage_data = response.get("usage")
        if usage_data:
            usage = ModelUsage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return ModelOutput(
            model=response.get("model", self.model_name),
            choices=choices,
            usage=usage,
        )

    def _map_finish_reason(self, reason: str | None) -> StopReason:
        """Map API finish reason to StopReason."""
        return as_stop_reason(reason)

    @override
    def should_retry(self, ex: BaseException) -> bool:
        """Determine if the request should be retried."""
        if isinstance(ex, httpx.HTTPStatusError):
            # Retry on 429 (rate limit) and 5xx errors
            return ex.response.status_code == 429 or ex.response.status_code >= 500
        if isinstance(ex, httpx.RequestError):
            # Retry on connection errors
            return True
        return False

    @override
    def is_auth_failure(self, ex: Exception) -> bool:
        """Check if the exception is an authentication failure."""
        if isinstance(ex, httpx.HTTPStatusError):
            return ex.response.status_code == 401
        return False

    @override
    def connection_key(self) -> str:
        """Return the connection key for rate limiting."""
        return self.base_url or "mockrest"
