import json
import os
import boto3
import time
import statistics
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dotenv import load_dotenv
from utils.utils import get_env_var

load_dotenv()
# Initialize clients based on provider
embedding_client = None
# Define the embedding model configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or "OpenAI"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER") or "OpenAI"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Default for OpenAI
LLM_MODEL = os.getenv("PRIMARY_MODEL") or "gpt-4o-mini"
aws_region = os.getenv("AWS_REGION") or "us-west-2"
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")
profile_name = os.getenv("AWS_PROFILE")


# Define model configurations
NOVA_MODELS = {
    "us.amazon.nova-pro-v1:0": {"context_length": 300000},
    "us.amazon.nova-micro-v1:0": {"context_length": 128000},
    "us.amazon.nova-lite-v1:0": {"context_length": 300000},
}


CLAUDE_MODELS = {
    "anthropic.claude-3-5-haiku-20241022-v1:0": {"context_length": 200000},
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"context_length": 200000},
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {"context_length": 200000},
}


if EMBEDDING_PROVIDER == "Bedrock" or LLM_PROVIDER == "Bedrock":
    # Initialize Bedrock client
    # session = boto3.Session(
    #     profile_name=os.getenv("AWS_PROFILE"), region_name=os.getenv("AWS_REGION")
    # )
    print(aws_access_key_id)
    print(aws_secret_access_key)
    print(aws_session_token)
    print(profile_name)
    print(aws_region)
    session = boto3.Session(
        aws_access_key_id,
        aws_secret_access_key,
        aws_session_token,
        region_name=aws_region,
    )
    bedrock_runtime = session.client("bedrock-runtime")
    if EMBEDDING_PROVIDER == "Bedrock":
        EMBEDDING_DIMENSION = 1024  # Titan embedding dimension


async def get_title_and_summary(chunk: str, url: str) -> Tuple[Dict[str, str], float]:
    """Extract title and summary using the configured LLM provider.
    Returns tuple of (result, inference_time_seconds)"""
    try:
        if LLM_PROVIDER == "Bedrock":
            model_id = LLM_MODEL
            start_time = time.time()

            # Common system prompt content
            system_content = """You are an AI that extracts titles and summaries from documentation chunks.
            You must respond with a JSON object containing 'title' and 'summary' keys.
            For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
            For the summary: Create a concise summary of the main points in this chunk.
            Keep both title and summary concise but informative.
            Example response format:
            {
                "title": "The extracted or derived title",
                "summary": "A concise summary of the content"
            }"""

            if model_id in NOVA_MODELS:
                # Nova models request format (unchanged)
                request_body = {
                    "schemaVersion": "messages-v1",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                            ],
                        }
                    ],
                    "system": [{"text": system_content}],
                    "inferenceConfig": {
                        "maxTokens": 1000,
                        "topP": 0.9,
                        "topK": 20,
                        "temperature": 0.7,
                    },
                }
            elif model_id in CLAUDE_MODELS:
                # Claude models request format
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "anthropic_beta": ["computer-use-2024-10-22"],
                    "max_tokens": 1000,
                    "system": system_content,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Please analyze this content and respond with a JSON object:\n\nURL: {url}\n\nContent:\n{chunk[:1000]}...",
                                }
                            ],
                        }
                    ],
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            else:
                raise ValueError(f"Unsupported model: {model_id}")

            response = bedrock_runtime.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            # Process the response stream
            full_response = ""
            stream = response.get("body")
            if stream:
                for event in stream:
                    chunk = json.loads(event.get("chunk").get("bytes").decode())
                    if model_id in NOVA_MODELS:
                        content_block_delta = chunk.get("contentBlockDelta")
                        if content_block_delta:
                            full_response += content_block_delta.get("delta", {}).get(
                                "text", ""
                            )
                    else:  # Claude models
                        delta = chunk.get("delta", {}).get("text", "")
                        if delta:
                            full_response += delta

            inference_time = time.time() - start_time

            try:
                # Debug print
                print(f"\nRaw response from {model_id}:\n{full_response}\n")
                print(f"Inference time: {inference_time:.2f} seconds")

                # Clean up the response
                full_response = full_response.strip()

                # Handle responses wrapped in ```json code blocks
                if full_response.startswith("```json"):
                    full_response = full_response.replace("```json", "").replace(
                        "```", ""
                    )

                # Find the first '{' and last '}'
                start_idx = full_response.find("{")
                end_idx = full_response.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = full_response[start_idx:end_idx]
                    return json.loads(json_str), inference_time
                else:
                    print(f"Could not find valid JSON markers in response")
                    return {
                        "title": "Error: No JSON found",
                        "summary": "Could not find valid JSON in response",
                    }, inference_time

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Attempted to parse: {full_response}")
                return {
                    "title": "Error processing JSON response",
                    "summary": "Failed to parse response as JSON",
                }, inference_time

    except Exception as e:
        print(f"Error getting title and summary: {str(e)}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary",
        }, 0.0


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from the configured provider (OpenAI or Bedrock)."""
    try:
        if EMBEDDING_PROVIDER == "Bedrock":
            embedding_request = json.dumps(
                {"inputText": text, "dimensions": EMBEDDING_DIMENSION}
            )
            response = bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=embedding_request,
                contentType="application/json",
                accept="application/json",
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
        return [0] * EMBEDDING_DIMENSION


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
    session = boto3.Session(profile_name=aws_profile, region_name=region)
    bedrock_runtime = session.client("bedrock-runtime")

    truncated_text = text[:8192]
    embedding_request = json.dumps(
        {
            "inputText": truncated_text,
            "dimensions": 1024,
        }
    )

    try:
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=embedding_request,
            contentType="application/json",
            accept="application/json",
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        return response_body["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0.0] * 1024


if __name__ == "__main__":

    async def main():
        text = "This is a sample text to generate an embedding for."
        url = "https://example.com"

        # Store timing results
        model_timings = {}

        # Test with Nova models
        print("\n=== Testing Nova Models ===")
        for nova_model in NOVA_MODELS:
            print(f"\nTesting with {nova_model}")
            model_timings[nova_model] = []

            # Run multiple times to get average
            num_runs = 3
            for i in range(num_runs):
                print(f"\nRun {i+1}/{num_runs}")
                global LLM_MODEL
                LLM_MODEL = nova_model
                result, inference_time = await get_title_and_summary(text, url)
                model_timings[nova_model].append(inference_time)
                print(f"Result: {result}")

        # Test with Claude models
        print("\n=== Testing Claude Models ===")
        for claude_model in CLAUDE_MODELS:
            print(f"\nTesting with {claude_model}")
            model_timings[claude_model] = []

            # Run multiple times to get average
            for i in range(num_runs):
                print(f"\nRun {i+1}/{num_runs}")
                LLM_MODEL = claude_model
                result, inference_time = await get_title_and_summary(text, url)
                model_timings[claude_model].append(inference_time)
                print(f"Result: {result}")

        # Print timing statistics
        print("\n=== Inference Time Statistics ===")
        for model, times in model_timings.items():
            avg_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)
            print(f"\n{model}:")
            print(f"  Average: {avg_time:.2f} seconds")
            print(f"  Std Dev: {std_dev:.2f} seconds")
            print(f"  Min: {min_time:.2f} seconds")
            print(f"  Max: {max_time:.2f} seconds")

    import asyncio

    asyncio.run(main())
