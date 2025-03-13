"""
File Connector for AiMee Knowledge Base.

This module provides a connector for importing knowledge from text files
into the AiMee knowledge base.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from knowledge.connectors.base_connector import BaseConnector

logger = logging.getLogger(__name__)

class FileConnector(BaseConnector):
    """
    Connector for importing knowledge from text files.
    
    This connector can import knowledge from:
    - Individual text files
    - Directories of text files
    - Files with specific extensions
    """
    
    def __init__(
        self,
        namespace: str = "default",
        config: Optional[Dict[str, Any]] = None,
        base_path: Optional[Union[str, Path]] = None,
        file_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize the file connector.
        
        Args:
            namespace: Namespace for the knowledge base
            config: Optional configuration parameters
            base_path: Base path for file operations
            file_extensions: List of file extensions to process (e.g., ['.txt', '.md'])
        """
        super().__init__(namespace, config)
        
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.file_extensions = file_extensions or ['.txt', '.md', '.csv', '.json']
    
    async def _initialize_connector(self) -> bool:
        """
        Initialize connector-specific resources.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        # Check if the base path exists
        if not self.base_path.exists():
            logger.error(f"Base path does not exist: {self.base_path}")
            return False
        
        logger.info(f"File connector initialized with base path: {self.base_path}")
        return True
    
    async def fetch_knowledge(
        self,
        query: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        recursive: bool = False,
        chunk_size: int = 1000,
        overlap: int = 200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Fetch knowledge from text files.
        
        Args:
            query: Optional glob pattern to filter files
            path: Path to a file or directory (relative to base_path)
            recursive: Whether to search directories recursively
            chunk_size: Size of text chunks to create
            overlap: Overlap between chunks
            **kwargs: Additional arguments
            
        Returns:
            List of knowledge items
        """
        if not self._initialized:
            raise RuntimeError("Connector not initialized. Call initialize() first.")
        
        # Determine the path to search
        search_path = self.base_path
        if path:
            search_path = self.base_path / path
        
        # Check if the path exists
        if not search_path.exists():
            logger.error(f"Path does not exist: {search_path}")
            return []
        
        # Get the list of files to process
        files_to_process = []
        
        if search_path.is_file():
            # Single file
            files_to_process.append(search_path)
        else:
            # Directory
            glob_pattern = query or "*"
            
            if recursive:
                # Recursive search
                for ext in self.file_extensions:
                    files_to_process.extend(search_path.glob(f"**/{glob_pattern}{ext}"))
            else:
                # Non-recursive search
                for ext in self.file_extensions:
                    files_to_process.extend(search_path.glob(f"{glob_pattern}{ext}"))
        
        # Process the files
        knowledge_items = []
        
        for file_path in files_to_process:
            try:
                # Read the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create metadata
                metadata = {
                    "file_path": str(file_path.relative_to(self.base_path)),
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix,
                    "file_size": file_path.stat().st_size,
                    "modified_time": file_path.stat().st_mtime,
                }
                
                # Split content into chunks if it's too large
                if len(content) > chunk_size:
                    chunks = self._split_text(content, chunk_size, overlap)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_index"] = i
                        chunk_metadata["total_chunks"] = len(chunks)
                        
                        knowledge_items.append({
                            "content": chunk,
                            "metadata": chunk_metadata,
                        })
                else:
                    # Add the whole content as a single item
                    knowledge_items.append({
                        "content": content,
                        "metadata": metadata,
                    })
                
                logger.info(f"Processed file: {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Fetched {len(knowledge_items)} knowledge items from {len(files_to_process)} files")
        return knowledge_items
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get the chunk
            end = start + chunk_size
            chunk = text[start:end]
            
            # Adjust to end at a sentence or paragraph if possible
            if end < len(text):
                # Try to find a paragraph break
                para_break = chunk.rfind('\n\n')
                if para_break != -1 and para_break > chunk_size // 2:
                    end = start + para_break + 2
                    chunk = text[start:end]
                else:
                    # Try to find a sentence break
                    sentence_breaks = [chunk.rfind('. '), chunk.rfind('! '), chunk.rfind('? ')]
                    sentence_break = max(sentence_breaks)
                    
                    if sentence_break != -1 and sentence_break > chunk_size // 2:
                        end = start + sentence_break + 2
                        chunk = text[start:end]
            
            chunks.append(chunk)
            
            # Move to the next chunk with overlap
            start = end - overlap
        
        return chunks
    
    async def close(self) -> None:
        """
        Close the connector and release any resources.
        """
        # No resources to close for this connector
        logger.info("File connector closed")

# Factory function to create a file connector
def create_file_connector(
    namespace: str = "default",
    base_path: Optional[Union[str, Path]] = None,
    file_extensions: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> FileConnector:
    """
    Create a file connector.
    
    Args:
        namespace: Namespace for the knowledge base
        base_path: Base path for file operations
        file_extensions: List of file extensions to process
        config: Optional configuration parameters
        
    Returns:
        File connector instance
    """
    return FileConnector(
        namespace=namespace,
        base_path=base_path,
        file_extensions=file_extensions,
        config=config,
    ) 