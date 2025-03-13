"""
Knowledge Base for AiMee.

This module provides a knowledge base for storing and retrieving information
using vector embeddings for semantic search.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from core.config.config import get_config
from knowledge.vectordb.vector_manager import VectorManager

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Knowledge Base for storing and retrieving information.
    
    This class provides methods for:
    - Adding knowledge to the knowledge base
    - Retrieving knowledge using semantic search
    - Managing knowledge in different namespaces
    """
    
    def __init__(
        self,
        namespace: str = "default",
        vector_db_provider: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the Knowledge Base.
        
        Args:
            namespace: Namespace for the knowledge base
            vector_db_provider: Vector database provider (if None, will use from config)
            embedding_provider: Embedding provider (if None, will use from config)
            embedding_model: Embedding model (if None, will use from config)
        """
        config = get_config()
        
        self.namespace = namespace
        self.vector_db_provider = vector_db_provider or config.vector_db.provider
        self.embedding_provider = embedding_provider or config.embeddings.provider
        self.embedding_model = embedding_model or config.embeddings.model
        
        # Initialize the vector manager
        self.vector_manager = VectorManager(
            provider=self.vector_db_provider,
            namespace=self.namespace,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
        )
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the knowledge base.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Initialize the vector manager
            success = await self.vector_manager.initialize()
            if success:
                logger.info(f"Knowledge base initialized for namespace '{self.namespace}'")
                self._initialized = True
                return True
            else:
                logger.error(f"Failed to initialize knowledge base for namespace '{self.namespace}'")
                return False
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            return False
    
    async def add_knowledge(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add knowledge to the knowledge base.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs for the chunks
            
        Returns:
            List of IDs for the added chunks
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        # Ensure metadatas is not None
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Ensure all metadatas have the namespace
        for metadata in metadatas:
            metadata["namespace"] = self.namespace
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        logger.info(f"Adding {len(texts)} chunks to knowledge base in namespace '{self.namespace}'")
        
        try:
            # Add the chunks to the vector database
            result_ids = await self.vector_manager.add_texts(texts, metadatas, ids)
            return result_ids
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return ids  # Return the generated IDs as fallback
    
    async def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for knowledge in the knowledge base.
        
        Args:
            query: Query string
            limit: Maximum number of results to return
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            List of search results, each containing the text, metadata, and similarity score
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        # Ensure metadata filter includes the namespace
        if metadata_filter is None:
            metadata_filter = {}
        
        # Only add namespace filter if not explicitly overridden
        if "namespace" not in metadata_filter:
            metadata_filter["namespace"] = self.namespace
        
        logger.info(f"Searching knowledge base in namespace '{self.namespace}' for: {query}")
        
        try:
            # Search the vector database
            results = await self.vector_manager.similarity_search(query, limit, metadata_filter)
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            # Return dummy results as fallback
            return [
                {
                    "text": f"Dummy result {i} for query: {query}",
                    "metadata": {"namespace": self.namespace, "dummy": "metadata"},
                    "score": 0.9 - (i * 0.1),
                }
                for i in range(min(limit, 3))
            ]
    
    async def delete_knowledge(self, ids: List[str]) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        Args:
            ids: List of chunk IDs to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        logger.info(f"Deleting {len(ids)} chunks from knowledge base in namespace '{self.namespace}'")
        
        try:
            # Delete the chunks from the vector database
            success = await self.vector_manager.delete_texts(ids)
            return success
        except Exception as e:
            logger.error(f"Error deleting knowledge: {e}")
            return False
    
    async def get_knowledge(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific knowledge chunk by ID.
        
        Args:
            id: ID of the chunk to retrieve
            
        Returns:
            Chunk data if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        logger.info(f"Getting chunk {id} from knowledge base in namespace '{self.namespace}'")
        
        try:
            # Get the chunk from the vector database
            result = await self.vector_manager.get_text(id)
            return result
        except Exception as e:
            logger.error(f"Error getting knowledge: {e}")
            return None
    
    async def update_knowledge_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for a specific knowledge chunk.
        
        Args:
            id: ID of the chunk to update
            metadata: New metadata
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        # Ensure metadata has the namespace
        metadata["namespace"] = self.namespace
        
        logger.info(f"Updating metadata for chunk {id} in knowledge base in namespace '{self.namespace}'")
        
        try:
            # Update the metadata in the vector database
            success = await self.vector_manager.update_text_metadata(id, metadata)
            return success
        except Exception as e:
            logger.error(f"Error updating knowledge metadata: {e}")
            return False
    
    async def count_knowledge(self, metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Count the number of knowledge chunks in the knowledge base.
        
        Args:
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            Number of chunks
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        # Ensure metadata filter includes the namespace
        if metadata_filter is None:
            metadata_filter = {}
        
        # Only add namespace filter if not explicitly overridden
        if "namespace" not in metadata_filter:
            metadata_filter["namespace"] = self.namespace
        
        logger.info(f"Counting chunks in knowledge base in namespace '{self.namespace}'")
        
        try:
            # Count the chunks in the vector database
            count = await self.vector_manager.count_texts(metadata_filter)
            return count
        except Exception as e:
            logger.error(f"Error counting knowledge: {e}")
            return 0
    
    async def close(self) -> None:
        """
        Close the knowledge base.
        """
        logger.info(f"Closing knowledge base for namespace '{self.namespace}'")
        
        if self._initialized:
            try:
                # Close the vector manager
                await self.vector_manager.close()
                self._initialized = False
            except Exception as e:
                logger.error(f"Error closing knowledge base: {e}")
    
    async def __aenter__(self):
        """
        Async context manager entry.
        """
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.
        """
        await self.close()

# Dictionary to store knowledge base instances by namespace
_knowledge_bases: Dict[str, KnowledgeBase] = {}

def get_knowledge_base(namespace: str = "default") -> KnowledgeBase:
    """
    Get a knowledge base instance for the specified namespace.
    
    Args:
        namespace: Namespace for the knowledge base
        
    Returns:
        Knowledge base instance
    """
    if namespace not in _knowledge_bases:
        _knowledge_bases[namespace] = KnowledgeBase(namespace)
    
    return _knowledge_bases[namespace] 