from __future__ import annotations as _annotations

import sys
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
from typing import Optional, List, Dict, Any

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var
from archon.agent_prompts import primary_coder_prompt
from archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool,
)

load_dotenv()

provider = get_env_var("PROVIDER") or "OpenAI"
llm = get_env_var("PRIMARY_MODEL") or "gpt-4o-mini"
base_url = get_env_var("BASE_URL") or "https://api.openai.com/v1"
api_key = get_env_var("LLM_API_KEY") or "no-llm-api-key-provided"
aws_region = get_env_var("AWS_REGION")
aws_access_key = get_env_var("AWS_ACCESS_KEY_ID")
aws_secret_key = get_env_var("AWS_SECRET_ACCESS_KEY")
aws_secret_key = get_env_var("AWS_SECRET_ACCESS_KEY")
aws_session_token = get_env_var("AWS_SESSION_TOKEN")


def get_model():
    if provider == "Bedrock":
        bedrock_provider_args = {
            "region_name": aws_region,
            "aws_access_key_id": aws_access_key,
            "aws_secret_access_key": aws_secret_key,
        }
        if aws_session_token:
            bedrock_provider_args["aws_session_token"] = aws_session_token
        return BedrockConverseModel(
            llm,  # e.g., 'anthropic.claude-3-sonnet-20240229-v1:0'
            provider=BedrockProvider(**bedrock_provider_args),
        )
    else:  # Default to OpenAI
        return OpenAIModel(llm, provider="openai")


model = get_model()

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class PydanticAIDeps:
    supabase: Client
    embedding_client: AsyncOpenAI
    reasoner_output: str


pydantic_ai_coder = Agent(
    model, system_prompt=primary_coder_prompt, deps_type=PydanticAIDeps, retries=2
)


@pydantic_ai_coder.system_prompt
def add_reasoner_output(ctx: RunContext[str]) -> str:
    return f"""
    \n\nAdditional thoughts/instructions from the reasoner LLM. 
    This scope includes documentation pages for you to search as well: 
    {ctx.deps.reasoner_output}
    """


@pydantic_ai_coder.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query

    Returns:
        A formatted string containing the top 4 most relevant documentation chunks
    """
    return await retrieve_relevant_documentation_tool(
        ctx.deps.supabase, ctx.deps.embedding_client, user_query
    )


@pydantic_ai_coder.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    return await list_documentation_pages_tool(ctx.deps.supabase)


@pydantic_ai_coder.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    return await get_page_content_tool(ctx.deps.supabase, url)
