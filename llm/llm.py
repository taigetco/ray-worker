from typing import Dict, Optional, List
import logging
import asyncio
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from ray import serve
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.usage.usage_lib import UsageContext

logger = logging.getLogger("ray.serve")

app = FastAPI()

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args, usage_context=UsageContext.OPENAI_API_SERVER)

        if engine_args.served_model_name is not None:
            self.served_model_names = engine_args.served_model_name
        else:
            self.served_model_names = [engine_args.model]

        self.model_config = None

        asyncio.create_task(self.load_model_config(response_role, lora_modules, chat_template))

        self.openai_serving_chat = None
        self.openai_serving_completion = None
        self.openai_serving_embedding = None

    async def load_model_config(self, response_role: str,
                                lora_modules: Optional[List[LoRAModulePath]] = None,
                                chat_template: Optional[str] = None):
        model_config = await self.engine.get_model_config()
        self.model_config = model_config
        self.openai_serving_chat = OpenAIServingChat(self.engine, model_config,
                                                     self.served_model_names,
                                                     response_role,
                                                     lora_modules,
                                                     chat_template)

        self.openai_serving_completion = OpenAIServingCompletion(self.engine, model_config,
                                                                 self.served_model_names, lora_modules)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        if self.openai_serving_chat is None:
            return JSONResponse(content={"error": "Model not loaded yet"}, status_code=503)

        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    parser = make_arg_parser()
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args

def build_llm_app(args: Dict[str, str]) -> serve.Application:
    parsed_args = parse_vllm_args(args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size

    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
    )