from __future__ import annotations as _annotations

import logfire
import os
import sys
from pydantic_ai import Agent
from dotenv import load_dotenv
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.bedrock import BedrockProvider
from supabase import Client

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.utils import get_env_var
from archon.agent_prompts import prompt_refiner_prompt

load_dotenv()

provider = get_env_var("LLM_PROVIDER") or "OpenAI"
llm = get_env_var("PRIMARY_MODEL") or "gpt-4o-mini"
base_url = get_env_var("BASE_URL") or "https://api.openai.com/v1"
api_key = get_env_var("LLM_API_KEY") or "no-llm-api-key-provided"
aws_region = get_env_var("AWS_REGION") or "us-west-2"

# Enhanced model initialization with Bedrock support
model = None
if provider == "Anthropic":
    model = AnthropicModel(llm, provider=AnthropicProvider(api_key=api_key))
elif provider == "Bedrock":
    model = BedrockConverseModel(
        llm,
        provider=BedrockProvider(
            region_name=aws_region,
            aws_access_key_id=get_env_var("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_env_var("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=get_env_var("AWS_SESSION_TOKEN"),
        ),
    )
else:  # Default to OpenAI
    model = OpenAIModel(
        llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key)
    )

logfire.configure(send_to_logfire="if-token-present")

prompt_refiner_agent = Agent(model, system_prompt=prompt_refiner_prompt, retries=2)
