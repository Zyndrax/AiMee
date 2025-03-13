"""
OpenAI Embeddings for AiMee.

This module provides integration with OpenAI's embedding models.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import openai
from openai import OpenAI

from core.config.config import get_config

logger = logging.getLogger(__name__)

class OpenAIEmbeddings:
    """
    OpenAI embeddings implementation.
    
    This class provides methods for generating embeddings using OpenAI's models.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 16,
    ):
        """
        Initialize the OpenAI embeddings.
        
        Args:
            api_key: OpenAI API key (if None, will use from config)
            organization: OpenAI organization (if None, will use from config)
            model: Embedding model to use (if None, will use from config)
            dimensions: Embedding dimensions (if None, will use default for model)
            batch_size: Batch size for embedding requests
        """
        config = get_config()
        
        self.api_key = api_key or config.ai_models.openai.api_key
        self.organization = organization or config.ai_models.openai.organization
        self.model = model or "text-embedding-3-small"
        self.dimensions = dimensions
        self.batch_size = batch_size
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either directly or via environment variables")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization,
        )
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Process in batches to avoid rate limits
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
            
            logger.debug(f"Embedded batch {i // self.batch_size + 1}/{(len(texts) + self.batch_size - 1) // self.batch_size}")
        
        return embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        # For OpenAI, query and document embeddings use the same method
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: Batch of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            # Create embedding request
            kwargs = {
                "model": self.model,
                "input": texts,
            }
            
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            
            # Make the API call
            response = await self._make_embedding_request(**kwargs)
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            dimension = self.dimensions or 1536  # Default for OpenAI embeddings
            return [[0.0] * dimension for _ in texts]
    
    async def _make_embedding_request(self, **kwargs) -> Any:
        """
        Make an embedding request to the OpenAI API.
        
        Args:
            **kwargs: Keyword arguments for the embedding request
            
        Returns:
            API response
        """
        # This is a synchronous call wrapped in an async function
        # In a production environment, you might want to use a proper async client
        return self.client.embeddings.create(**kwargs) 