from typing import List
from pydantic_ai.models.bedrock import LatestBedrockModelNames


def get_available_bedrock_models() -> List[str]:
    """Get list of available Bedrock models."""
    return list(LatestBedrockModelNames.__args__)


def validate_bedrock_model(model_name: str) -> bool:
    """Validate if the model name is a valid Bedrock model."""
    available_models = get_available_bedrock_models()
    return model_name in available_models


def validate_model_configuration(provider: str, model_name: str) -> bool:
    """Validate model configuration based on provider."""
    if provider == "Bedrock":
        return validate_bedrock_model(model_name)
    # Add other provider validations here
    return True
