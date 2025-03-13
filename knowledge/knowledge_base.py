"""
Knowledge Base for AiMee.

This module provides a knowledge base that uses the vector database to store and retrieve knowledge.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from knowledge.vectordb.vector_manager import get_vector_manager

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Knowledge base for storing and retrieving information.
    
    This class provides methods for:
    - Adding knowledge to the knowledge base
    - Retrieving knowledge from the knowledge base
    - Managing knowledge in the knowledge base
    """
    
    def __init__(self, namespace: str = "default"):
        """
        Initialize the knowledge base.
        
        Args:
            namespace: Namespace for the knowledge base, used to separate different knowledge domains
        """
        self.namespace = namespace
        self.vector_manager = get_vector_manager()
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the knowledge base.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if not self._initialized:
                success = await self.vector_manager.initialize()
                if success:
                    self._initialized = True
                    logger.info(f"Knowledge base '{self.namespace}' initialized successfully")
                    return True
                else:
                    logger.error(f"Failed to initialize knowledge base '{self.namespace}'")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error initializing knowledge base '{self.namespace}': {e}")
            return False
    
    async def add_knowledge(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """
        Add knowledge to the knowledge base.
        
        Args:
            texts: Text or list of texts to add
            metadata: Optional metadata or list of metadata dictionaries
            ids: Optional ID or list of IDs for the texts
            
        Returns:
            List of IDs for the added texts
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Convert single metadata to list
        if metadata is not None and not isinstance(metadata, list):
            metadata = [metadata]
        
        # Convert single ID to list
        if ids is not None and not isinstance(ids, list):
            ids = [ids]
        
        # Add namespace to metadata
        if metadata is None:
            metadata = [{"namespace": self.namespace} for _ in texts]
        else:
            for meta in metadata:
                meta["namespace"] = self.namespace
        
        return await self.vector_manager.add_texts(
            texts=texts,
            metadatas=metadata,
            ids=ids,
        )
    
    async def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        additional_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for knowledge in the knowledge base.
        
        Args:
            query: Query text
            limit: Maximum number of results to return
            additional_filter: Optional additional filter for metadata fields
            
        Returns:
            List of search results, each containing the content, metadata, and similarity score
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        # Create metadata filter with namespace
        metadata_filter = {"namespace": self.namespace}
        
        # Add additional filter if provided
        if additional_filter:
            metadata_filter.update(additional_filter)
        
        return await self.vector_manager.search(
            query=query,
            limit=limit,
            metadata_filter=metadata_filter,
        )
    
    async def delete_knowledge(self, ids: Union[str, List[str]]) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        Args:
            ids: ID or list of IDs to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        # Convert single ID to list
        if isinstance(ids, str):
            ids = [ids]
        
        return await self.vector_manager.delete(ids)
    
    async def get_knowledge(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific knowledge item by ID.
        
        Args:
            id: ID of the knowledge item to retrieve
            
        Returns:
            Knowledge item if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        return await self.vector_manager.get(id)
    
    async def update_knowledge_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for a specific knowledge item.
        
        Args:
            id: ID of the knowledge item to update
            metadata: New metadata
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        # Ensure namespace is preserved
        metadata["namespace"] = self.namespace
        
        return await self.vector_manager.update_metadata(id, metadata)
    
    async def count_knowledge(self, additional_filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Count the number of knowledge items in the knowledge base.
        
        Args:
            additional_filter: Optional additional filter for metadata fields
            
        Returns:
            Number of knowledge items
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        
        # Create metadata filter with namespace
        metadata_filter = {"namespace": self.namespace}
        
        # Add additional filter if provided
        if additional_filter:
            metadata_filter.update(additional_filter)
        
        return await self.vector_manager.count(metadata_filter)
    
    async def close(self) -> None:
        """
        Close the knowledge base.
        """
        if self._initialized:
            # We don't close the vector manager here because it might be shared with other knowledge bases
            self._initialized = False
            logger.info(f"Knowledge base '{self.namespace}' closed")

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