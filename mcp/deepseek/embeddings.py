"""
Deepseek Embeddings for AiMee.

This module provides integration with Deepseek's embedding models.
"""

import logging
import requests
from typing import Any, Dict, List, Optional, Union

from core.config.config import get_config

logger = logging.getLogger(__name__)

class DeepseekEmbeddings:
    """
    Deepseek embeddings implementation.
    
    This class provides methods for generating embeddings using Deepseek's models.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 16,
    ):
        """
        Initialize the Deepseek embeddings.
        
        Args:
            api_key: Deepseek API key (if None, will use from config)
            model: Embedding model to use (if None, will use from config)
            api_base: Deepseek API base URL (if None, will use from config)
            dimensions: Embedding dimensions (if None, will use default for model)
            batch_size: Batch size for embedding requests
        """
        config = get_config()
        
        self.api_key = api_key or config.ai_models.deepseek.api_key
        self.model = model or config.ai_models.deepseek.embedding_model or "deepseek-ai/deepseek-embedding-v1"
        self.api_base = api_base or config.ai_models.deepseek.api_base or "https://api.deepseek.com"
        self.dimensions = dimensions or config.ai_models.deepseek.embedding_dimensions or 1536
        self.batch_size = batch_size
        
        if not self.api_key:
            raise ValueError("Deepseek API key must be provided either directly or via environment variables")
    
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
        # For Deepseek, query and document embeddings use the same method
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
            embeddings = []
            
            for text in texts:
                # Make the API call for each text
                embedding = await self._make_embedding_request(text)
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with Deepseek: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * self.dimensions for _ in texts]
    
    async def _make_embedding_request(self, text: str) -> List[float]:
        """
        Make an embedding request to the Deepseek API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": text,
                "dimensions": self.dimensions
            }
            
            # This is a synchronous call wrapped in an async function
            # In a production environment, you might want to use a proper async client
            response = requests.post(
                f"{self.api_base}/v1/embeddings",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Deepseek API error: {response.status_code} - {response.text}")
                return [0.0] * self.dimensions
            
            result = response.json()
            embedding = result["data"][0]["embedding"]
            
            return embedding
        except Exception as e:
            logger.error(f"Error making Deepseek embedding request: {e}")
            return [0.0] * self.dimensions 