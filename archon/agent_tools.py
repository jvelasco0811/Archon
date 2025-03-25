from typing import Dict, Any, List, Optional, Union
from openai import AsyncOpenAI
from supabase import Client
import sys
import os
import json
import boto3
from boto3.session import Session
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var

# Configuration constants
embedding_model = get_env_var("EMBEDDING_MODEL") or "text-embedding-3-small"
embedding_provider = get_env_var("EMBEDDING_PROVIDER") or "OpenAI"
aws_region = get_env_var("AWS_REGION") or "us-west-2"
EMBEDDING_DIMENSIONS = {"OpenAI": 1536, "Bedrock": 1024}

# Initialize Bedrock client at module level
_bedrock_client: Optional[BedrockRuntimeClient] = None


async def get_bedrock_client() -> BedrockRuntimeClient:
    """
    Initialize and return a Bedrock client using boto3.
    Uses singleton pattern to avoid multiple client instantiations.

    Returns:
        BedrockRuntimeClient: Configured Bedrock runtime client
    """
    global _bedrock_client

    if _bedrock_client is None:
        session = Session(
            region_name=aws_region,
            aws_access_key_id=get_env_var("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_env_var("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=get_env_var("AWS_SESSION_TOKEN"),
        )
        _bedrock_client = session.client("bedrock-runtime")

    return _bedrock_client


async def get_bedrock_embedding(
    text: str, bedrock_client: BedrockRuntimeClient
) -> List[float]:
    """
    Get embedding vector from AWS Bedrock.

    Args:
        text: The text to embed
        bedrock_client: Configured Bedrock runtime client

    Returns:
        List[float]: The embedding vector
    """
    try:
        embedding_request = json.dumps(
            {"inputText": text, "dimensions": EMBEDDING_DIMENSIONS["Bedrock"]}
        )
        response = bedrock_client.invoke_model(
            modelId=embedding_model, body=embedding_request.encode()
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        return response_body["embedding"]
    except Exception as e:
        print(f"Error getting Bedrock embedding: {e}")
        return [0] * EMBEDDING_DIMENSIONS["Bedrock"]


async def get_embedding(
    text: str, embedding_client: Union[AsyncOpenAI, BedrockRuntimeClient]
) -> List[float]:
    """
    Get embedding vector from the configured provider (OpenAI or Bedrock).

    Args:
        text: The text to embed
        embedding_client: Either AsyncOpenAI client or Bedrock client

    Returns:
        List[float]: The embedding vector
    """
    try:
        if embedding_provider == "Bedrock":
            if not isinstance(embedding_client, BedrockRuntimeClient):
                embedding_client = await get_bedrock_client()
            return await get_bedrock_embedding(text, embedding_client)

        # OpenAI embedding
        response = await embedding_client.embeddings.create(
            model=embedding_model, input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        dimensions = EMBEDDING_DIMENSIONS.get(
            embedding_provider, EMBEDDING_DIMENSIONS["OpenAI"]
        )
        return [0] * dimensions


async def retrieve_relevant_documentation_tool(
    supabase: Client,
    embedding_client: Union[AsyncOpenAI, BedrockRuntimeClient],
    user_query: str,
) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.

    Args:
        supabase: Supabase client for database operations
        embedding_client: Embedding provider client (OpenAI or Bedrock)
        user_query: The user's question or query

    Returns:
        str: Formatted string containing relevant documentation chunks
    """
    try:
        query_embedding = await get_embedding(user_query, embedding_client)

        # Query Supabase for relevant documents
        result = supabase.rpc(
            "match_site_pages",
            {
                "query_embedding": query_embedding,
                "match_count": 4,
                "filter": {"source": "pydantic_ai_docs"},
            },
        ).execute()

        if not result.data:
            return "No relevant documentation found."

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


async def list_documentation_pages_tool(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available Pydantic AI documentation pages.
    This is called by the list_documentation_pages tool and also externally
    to fetch documentation pages for the reasoner LLM.

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = (
            supabase.from_("site_pages")
            .select("url")
            .eq("metadata->>source", "pydantic_ai_docs")
            .execute()
        )

        if not result.data:
            return []

        # Extract unique URLs
        urls = sorted(set(doc["url"] for doc in result.data))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


async def get_page_content_tool(supabase: Client, url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = (
            supabase.from_("site_pages")
            .select("title, content, chunk_number")
            .eq("url", url)
            .eq("metadata->>source", "pydantic_ai_docs")
            .order("chunk_number")
            .execute()
        )

        if not result.data:
            return f"No content found for URL: {url}"

        # Format the page with its title and all chunks
        page_title = result.data[0]["title"].split(" - ")[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk["content"])

        # Join everything together but limit the characters in case the page is massive (there are a coule big ones)
        # This will be improved later so if the page is too big RAG will be performed on the page itself
        return "\n\n".join(formatted_content)[:20000]

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
