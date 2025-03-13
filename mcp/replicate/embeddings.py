"""
Replicate Embeddings for AiMee.

This module provides integration with Replicate's embedding models.
"""

import logging
import replicate
import asyncio
from typing import Any, Dict, List, Optional, Union

from core.config.config import get_config

logger = logging.getLogger(__name__)

class ReplicateEmbeddings:
    """
    Replicate embeddings implementation.
    
    This class provides methods for generating embeddings using Replicate's models.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 8,
    ):
        """
        Initialize the Replicate embeddings.
        
        Args:
            api_key: Replicate API key (if None, will use from config)
            model: Embedding model to use (if None, will use from config)
                   Format should be "owner/model:version" or "owner/model"
            dimensions: Embedding dimensions (if None, will use default for model)
            batch_size: Batch size for embedding requests
        """
        config = get_config()
        
        self.api_key = api_key or config.ai_models.replicate.api_key
        self.model = model or config.ai_models.replicate.embedding_model
        self.dimensions = dimensions or config.ai_models.replicate.embedding_dimensions or 1536
        self.batch_size = batch_size
        
        if not self.api_key:
            raise ValueError("Replicate API key must be provided either directly or via environment variables")
        
        if not self.model:
            raise ValueError("Replicate model must be provided either directly or via environment variables")
        
        # Set the API key for the replicate client
        replicate.api_token = self.api_key
    
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
        # For Replicate, query and document embeddings use the same method
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
            # Create embedding requests for each text in the batch
            tasks = [self._make_embedding_request(text) for text in texts]
            
            # Run the requests concurrently
            embeddings = await asyncio.gather(*tasks)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with Replicate: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * self.dimensions for _ in texts]
    
    async def _make_embedding_request(self, text: str) -> List[float]:
        """
        Make an embedding request to the Replicate API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Parse the model string to get owner, model, and version
            if ":" in self.model:
                model_path, version = self.model.split(":")
            else:
                model_path = self.model
                version = None
            
            # Create the input for the model
            inputs = {
                "text": text,
            }
            
            # If dimensions are specified, add them to the inputs
            if self.dimensions:
                inputs["dimensions"] = self.dimensions
            
            # Run the model
            # This is a synchronous call wrapped in an async function
            # In a production environment, you might want to use a proper async client
            loop = asyncio.get_event_loop()
            if version:
                result = await loop.run_in_executor(
                    None, 
                    lambda: replicate.run(
                        f"{model_path}:{version}",
                        input=inputs
                    )
                )
            else:
                result = await loop.run_in_executor(
                    None, 
                    lambda: replicate.run(
                        model_path,
                        input=inputs
                    )
                )
            
            # Extract the embedding from the result
            # The exact format depends on the model, so we need to handle different cases
            if isinstance(result, list) and all(isinstance(x, float) for x in result):
                # Result is already a list of floats
                embedding = result
            elif isinstance(result, dict) and "embedding" in result:
                # Result is a dict with an "embedding" key
                embedding = result["embedding"]
            elif isinstance(result, dict) and "data" in result and isinstance(result["data"], list):
                # Result is a dict with a "data" key containing a list
                embedding = result["data"][0]["embedding"]
            else:
                # Unknown format, log an error and return zeros
                logger.error(f"Unknown Replicate embedding result format: {result}")
                return [0.0] * self.dimensions
            
            return embedding
        except Exception as e:
            logger.error(f"Error making Replicate embedding request: {e}")
            return [0.0] * self.dimensions 