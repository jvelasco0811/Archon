import json
import os
import boto3
from typing import List
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Callable, Union

load_dotenv()
# Initialize clients based on provider
embedding_client = None
# Define the embedding model configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or "OpenAI"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER") or "OpenAI"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Default for OpenAI
LLM_MODEL = os.getenv("PRIMARY_MODEL") or "gpt-4o-mini"

if EMBEDDING_PROVIDER == "Bedrock" or LLM_PROVIDER == "Bedrock":
    # Initialize Bedrock client
    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE"), region_name=os.getenv("AWS_REGION")
    )
    bedrock_runtime = session.client("bedrock-runtime")
    if EMBEDDING_PROVIDER == "Bedrock":
        EMBEDDING_DIMENSION = 1024  # Titan embedding dimension


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using the configured LLM provider."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        if LLM_PROVIDER == "Bedrock":
            prompt = f"{system_prompt}\n\nURL: {url}\n\nContent:\n{chunk[:1000]}..."
            request_body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."
                                }
                            ]
                        }
                    ],
                    "temperature": 0.7,
                    "top_p": 0.999,
                    "top_k": 250,
                }
            )

            response = bedrock_runtime.invoke_model(
                modelId=LLM_MODEL or "anthropic.claude-3-5-haiku-20241022-v1:0",
                body=request_body,
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            return json.loads(response_body["content"][0]["text"])

    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary",
        }


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from the configured provider (OpenAI or Bedrock)."""
    try:
        if EMBEDDING_PROVIDER == "Bedrock":
            embedding_request = json.dumps(
                {"inputText": text, "dimensions": EMBEDDING_DIMENSION}
            )
            response = bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0", body=embedding_request
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body["embedding"]
        else:  # OpenAI
            response = await embedding_client.embeddings.create(
                model=EMBEDDING_MODEL, input=text
            )
            return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [
            0
        ] * EMBEDDING_DIMENSION  # Return zero vector matching the model dimension


def generate_embedding(
    text: str, aws_profile: str = "jaime.dev", region: str = "us-west-2"
) -> List[float]:
    """
    Generate embedding using Amazon Titan Embed Text v2

    Args:
        text: Input text to generate embedding for
        aws_profile: AWS credentials profile name (optional)
        region: AWS region name (defaults to us-west-2)

    Returns:
        List[float]: 1024-dimensional embedding vector

    Note:
        Max input tokens: 8192
        Vector size: 1024
    """
    # Initialize Bedrock client
    session = boto3.Session(profile_name=aws_profile, region_name=region)
    bedrock_runtime = session.client("bedrock-runtime")

    # Truncate content if it exceeds max tokens
    truncated_text = text[:8192]

    # Prepare request payload
    embedding_request = json.dumps(
        {
            "inputText": truncated_text,
            "dimensions": 1024,  # Explicitly specify 1024-dimensional embedding
        }
    )

    try:
        # Call Bedrock API
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v2:0", body=embedding_request
        )

        # Parse response
        response_body = json.loads(response["body"].read().decode("utf-8"))
        return response_body["embedding"]

    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * 1024


if __name__ == "__main__":
    async def main():
        text = "This is a sample text to generate an embedding for."
        url = "https://example.com"
        extracted = await get_title_and_summary(text, url)
        embedding = await get_embedding(text)
        # Get title and summary
        print(f"Generated embedding: {embedding}")
        print(f"Extracted title and summary: {extracted}")

    import asyncio
    asyncio.run(main())
