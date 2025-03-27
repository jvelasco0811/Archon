from __future__ import annotations as _annotations
from archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool,
)
from archon.agent_prompts import agent_refiner_prompt
from utils.utils import get_env_var
import os
import sys
from dataclasses import dataclass
import boto3
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from dotenv import load_dotenv
import logfire
import asyncio
import httpx

import json
from typing import List, Union
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from openai import AsyncOpenAI
from supabase import Client
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.bedrock import BedrockProvider

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

load_dotenv()

provider = get_env_var("LLM_PROVIDER") or "OpenAI"
llm = get_env_var("PRIMARY_MODEL") or "gpt-4o-mini"
base_url = get_env_var("BASE_URL") or "https://api.openai.com/v1"
api_key = get_env_var("LLM_API_KEY") or "no-llm-api-key-provided"
aws_region = get_env_var("AWS_REGION") or "us-west-2"
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")
profile_name = os.getenv("AWS_PROFILE")

# Enhanced model initialization with Bedrock support
model = None
# session = None
if provider == "Anthropic":
    model = AnthropicModel(llm, provider=AnthropicProvider(api_key=api_key))
elif provider == "Bedrock":
    # Initialize Bedrock client
    # print("Session created with keys")
    # print(session)
    # print(session.client("bedrock-runtime"))
    # print(aws_region)
    # print(aws_access_key_id)
    # print(aws_secret_access_key)
    # print(aws_session_token)
    # print(profile_name)
    # if (
    #     os.getenv("AWS_AUTH_METHOD") == "profile"
    #     and os.getenv("AWS_PROFILE") is not None
    # ):
    #     session = boto3.Session(profile_name, region_name=aws_region)

    # if (
    #     os.getenv("AWS_AUTH_METHOD") == "keys"
    #     and os.getenv("AWS_ACCESS_KEY_ID") is not None
    #     and os.getenv("AWS_SECRET_ACCESS_KEY") is not None
    # ):
    session = boto3.Session(
        aws_access_key_id,
        aws_secret_access_key,
        aws_session_token,
        region_name=aws_region,
    )

    # Insert a check to ensure the session was created.
    if session is None:
        raise ValueError(
            "AWS session not initialized. Check your AWS credentials and authentication method."
        )

    bedrock_client = session.client("bedrock-runtime")
    model = BedrockConverseModel(
        llm, provider=BedrockProvider(bedrock_client=bedrock_client)
    )
else:  # Default to OpenAI
    model = OpenAIModel(
        llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key)
    )

embedding_model = get_env_var("EMBEDDING_MODEL") or "text-embedding-3-small"

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class AgentRefinerDeps:
    supabase: Client
    embedding_client: Union[
        AsyncOpenAI, BedrockRuntimeClient
    ]  # Support both OpenAI and Bedrock clients


agent_refiner_agent = Agent(
    model, system_prompt=agent_refiner_prompt, deps_type=AgentRefinerDeps, retries=2
)


@agent_refiner_agent.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[AgentRefinerDeps], query: str
) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    Make sure your searches always focus on implementing the agent itself.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        query: Your query to retrieve relevant documentation for implementing agents

    Returns:
        A formatted string containing the top 4 most relevant documentation chunks
    """
    return await retrieve_relevant_documentation_tool(
        ctx.deps.supabase, ctx.deps.embedding_client, query
    )


@agent_refiner_agent.tool
async def list_documentation_pages(ctx: RunContext[AgentRefinerDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    This will give you all pages available, but focus on the ones related to configuring agents and their dependencies.

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    return await list_documentation_pages_tool(ctx.deps.supabase)


@agent_refiner_agent.tool
async def get_page_content(ctx: RunContext[AgentRefinerDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    Only use this tool to get pages related to setting up agents with Pydantic AI.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    return await get_page_content_tool(ctx.deps.supabase, url)
