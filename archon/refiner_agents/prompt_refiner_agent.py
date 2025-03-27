from __future__ import annotations as _annotations
from archon.agent_prompts import prompt_refiner_prompt
from utils.utils import get_env_var
import boto3
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
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
)

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
    # Initialize Bedrock client
    session = None
    if (os.getenv("AWS_AUTH_METHOD") == "profile" and os.getenv("AWS_PROFILE") is not None):
        session = boto3.Session(
            profile_name=os.getenv("AWS_PROFILE"), region_name=os.getenv("AWS_REGION")
        )

    if (os.getenv("AWS_AUTH_METHOD") == "keys" and os.getenv("AWS_ACCESS_KEY_ID") is not None and os.getenv("AWS_SECRET_ACCESS_KEY") is not None):
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            region_name=os.getenv("AWS_REGION")
        )
    bedrock_client = session.client("bedrock-runtime")

    model = BedrockConverseModel(
        llm, provider=BedrockProvider(bedrock_client=bedrock_client)
    )

else:  # Default to OpenAI
    model = OpenAIModel(
        llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key)
    )

logfire.configure(send_to_logfire="if-token-present")

prompt_refiner_agent = Agent(
    model, system_prompt=prompt_refiner_prompt, retries=2)
