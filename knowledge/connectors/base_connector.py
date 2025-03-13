"""
Base Connector for AiMee Knowledge Base.

This module provides a base class for connectors that can be used to import knowledge
from external sources into the AiMee knowledge base.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from knowledge.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)

class BaseConnector(ABC):
    """
    Base class for knowledge connectors.
    
    Knowledge connectors are used to import knowledge from external sources
    into the AiMee knowledge base.
    """
    
    def __init__(self, namespace: str = "default", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the connector.
        
        Args:
            namespace: Namespace for the knowledge base
            config: Optional configuration parameters
        """
        self.namespace = namespace
        self.config = config or {}
        self.knowledge_base = get_knowledge_base(namespace)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the connector.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if not self._initialized:
                # Initialize the knowledge base
                kb_success = await self.knowledge_base.initialize()
                if not kb_success:
                    logger.error(f"Failed to initialize knowledge base for connector in namespace '{self.namespace}'")
                    return False
                
                # Initialize the connector-specific resources
                connector_success = await self._initialize_connector()
                if not connector_success:
                    logger.error(f"Failed to initialize connector-specific resources in namespace '{self.namespace}'")
                    return False
                
                self._initialized = True
                logger.info(f"Connector in namespace '{self.namespace}' initialized successfully")
                return True
            return True
        except Exception as e:
            logger.error(f"Error initializing connector in namespace '{self.namespace}': {e}")
            return False
    
    @abstractmethod
    async def _initialize_connector(self) -> bool:
        """
        Initialize connector-specific resources.
        
        This method should be implemented by subclasses to initialize any
        connector-specific resources.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def fetch_knowledge(self, query: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch knowledge from the external source.
        
        This method should be implemented by subclasses to fetch knowledge from
        the external source.
        
        Args:
            query: Optional query to filter the knowledge
            **kwargs: Additional arguments for the specific connector
            
        Returns:
            List of knowledge items, each as a dictionary with at least 'content' and 'metadata' keys
        """
        pass
    
    async def import_knowledge(self, query: Optional[str] = None, **kwargs) -> List[str]:
        """
        Import knowledge from the external source into the knowledge base.
        
        Args:
            query: Optional query to filter the knowledge
            **kwargs: Additional arguments for the specific connector
            
        Returns:
            List of IDs for the imported knowledge items
        """
        if not self._initialized:
            raise RuntimeError("Connector not initialized. Call initialize() first.")
        
        # Fetch knowledge from the external source
        knowledge_items = await self.fetch_knowledge(query, **kwargs)
        
        if not knowledge_items:
            logger.warning(f"No knowledge items fetched from connector in namespace '{self.namespace}'")
            return []
        
        # Extract content and metadata
        texts = []
        metadatas = []
        
        for item in knowledge_items:
            if "content" not in item:
                logger.warning(f"Skipping knowledge item without content: {item}")
                continue
            
            texts.append(item["content"])
            metadatas.append(item.get("metadata", {}))
        
        if not texts:
            logger.warning(f"No valid knowledge items to import in namespace '{self.namespace}'")
            return []
        
        # Add source information to metadata
        for metadata in metadatas:
            metadata["source"] = self.__class__.__name__
        
        # Add knowledge to the knowledge base
        return await self.knowledge_base.add_knowledge(texts=texts, metadata=metadatas)
    
    @abstractmethod
    async def close(self) -> None:
        """
        Close the connector and release any resources.
        
        This method should be implemented by subclasses to close any
        connector-specific resources.
        """
        pass 