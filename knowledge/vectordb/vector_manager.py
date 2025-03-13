"""
Vector Database Manager for AiMee.

This module provides a high-level interface for working with vector databases,
abstracting away the details of the underlying implementation.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from core.config.config import get_config
from knowledge.vectordb.supabase_vector import SupabaseVectorDB
from mcp.embeddings_factory import get_embeddings

logger = logging.getLogger(__name__)

class VectorManager:
    """
    Manager for vector database operations.
    
    This class provides a high-level interface for working with vector databases,
    abstracting away the details of the underlying implementation.
    """
    
    def __init__(
        self,
        vector_db: Optional[SupabaseVectorDB] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the vector manager.
        
        Args:
            vector_db: Optional vector database instance. If None, a new one will be created.
            embedding_provider: Provider for embeddings (e.g., 'openai', 'anthropic')
            embedding_model: Model name for embeddings
        """
        config = get_config()
        
        self.vector_db = vector_db or SupabaseVectorDB()
        self.embedding_provider = embedding_provider or config.ai_models.default_provider
        self.embedding_model = embedding_model
        self._initialized = False
        self._embeddings = None
    
    async def initialize(self) -> bool:
        """
        Initialize the vector manager.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if not self._initialized:
                # Initialize the vector database
                db_success = await self.vector_db.initialize()
                if not db_success:
                    logger.error("Failed to initialize vector database")
                    return False
                
                # Initialize the embeddings model
                try:
                    self._embeddings = get_embeddings(
                        provider=self.embedding_provider,
                        model=self.embedding_model,
                    )
                    logger.info(f"Using {self.embedding_provider} embeddings")
                except Exception as e:
                    logger.error(f"Failed to initialize embeddings model: {e}")
                    return False
                
                self._initialized = True
                logger.info("Vector manager initialized successfully")
                return True
            return True
        except Exception as e:
            logger.error(f"Error initializing vector manager: {e}")
            return False
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts to the vector database.
        
        If embeddings are not provided, they will be generated using the default embedding model.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            embeddings: Optional list of pre-computed embeddings
            ids: Optional list of IDs for the texts
            
        Returns:
            List of IDs for the added texts
        """
        if not self._initialized:
            raise RuntimeError("Vector manager not initialized. Call initialize() first.")
        
        if embeddings is None:
            # Generate embeddings using the default embedding model
            embeddings = await self._generate_embeddings(texts)
        
        return await self.vector_db.add_embeddings(
            embeddings=embeddings,
            contents=texts,
            metadatas=metadatas,
            ids=ids,
        )
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts in the vector database.
        
        Args:
            query: Query text
            limit: Maximum number of results to return
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            List of search results, each containing the content, metadata, and similarity score
        """
        if not self._initialized:
            raise RuntimeError("Vector manager not initialized. Call initialize() first.")
        
        # Generate embedding for the query
        query_embedding = await self._generate_embedding(query)
        
        return await self.vector_db.search_embeddings(
            query_embedding=query_embedding,
            limit=limit,
            metadata_filter=metadata_filter,
        )
    
    async def delete(self, ids: List[str]) -> bool:
        """
        Delete texts from the vector database.
        
        Args:
            ids: List of text IDs to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Vector manager not initialized. Call initialize() first.")
        
        return await self.vector_db.delete_embeddings(ids)
    
    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific text by ID.
        
        Args:
            id: ID of the text to retrieve
            
        Returns:
            Text data if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Vector manager not initialized. Call initialize() first.")
        
        return await self.vector_db.get_embedding(id)
    
    async def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for a specific text.
        
        Args:
            id: ID of the text to update
            metadata: New metadata
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Vector manager not initialized. Call initialize() first.")
        
        return await self.vector_db.update_embedding_metadata(id, metadata)
    
    async def count(self, metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Count the number of texts in the database.
        
        Args:
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            Number of texts
        """
        if not self._initialized:
            raise RuntimeError("Vector manager not initialized. Call initialize() first.")
        
        return await self.vector_db.count_embeddings(metadata_filter)
    
    async def close(self) -> None:
        """
        Close the vector manager.
        """
        if self._initialized:
            await self.vector_db.close()
            self._initialized = False
            logger.info("Vector manager closed")
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not self._embeddings:
            raise RuntimeError("Embeddings model not initialized")
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        return await self._embeddings.embed_documents(texts)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        if not self._embeddings:
            raise RuntimeError("Embeddings model not initialized")
        
        logger.info(f"Generating embedding for text: {text[:50]}...")
        return await self._embeddings.embed_query(text)

# Global vector manager instance
vector_manager = VectorManager()

def get_vector_manager() -> VectorManager:
    """Get the global vector manager instance."""
    return vector_manager 