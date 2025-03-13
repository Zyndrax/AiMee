"""
Embeddings Factory for AiMee.

This module provides a factory for creating embedding models.
"""

import logging
from typing import Any, Dict, Optional, Union

from core.config.config import get_config
from mcp.anthropic.embeddings import AnthropicEmbeddings
from mcp.openai.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingsFactory:
    """
    Factory for creating embedding models.
    
    This class provides methods for creating embedding models based on
    the provider and model name.
    """
    
    @staticmethod
    def create_embeddings(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[OpenAIEmbeddings, AnthropicEmbeddings]:
        """
        Create an embedding model.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            model: Model name
            **kwargs: Additional arguments for the embedding model
            
        Returns:
            Embedding model instance
            
        Raises:
            ValueError: If the provider is not supported
        """
        config = get_config()
        
        # Use default provider if not specified
        provider = provider or config.ai_models.default_provider
        
        # Create the embedding model based on the provider
        if provider.lower() == "openai":
            return OpenAIEmbeddings(model=model, **kwargs)
        elif provider.lower() == "anthropic":
            return AnthropicEmbeddings(model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

# Global embeddings factory instance
embeddings_factory = EmbeddingsFactory()

def get_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Union[OpenAIEmbeddings, AnthropicEmbeddings]:
    """
    Get an embedding model.
    
    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        model: Model name
        **kwargs: Additional arguments for the embedding model
        
    Returns:
        Embedding model instance
    """
    return embeddings_factory.create_embeddings(provider, model, **kwargs) 