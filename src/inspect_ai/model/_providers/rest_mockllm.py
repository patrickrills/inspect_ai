from inspect_ai.model import ModelAPI, GenerateConfig, ModelOutput
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai._util.constants import NO_CONTENT
import httpx


class RestMockLLM(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args,
    ):
        super().__init__(model_name, base_url, api_key, [], config)
        self.base_url = base_url or "http://localhost:3000"
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        response = await self._client.post(
            "/mock-llm-response",
            json={"input": input[0].content},
        )
        if response.status_code != 200:
            raise Exception(
                f"Error from Mock LLM API: {response.status_code} - {response.text}"
            )
        data = response.json()
        content = data["content"]
        return ModelOutput.from_content(self.model_name, content)
