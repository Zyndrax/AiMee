"""
File Connector for AiMee.

This module provides a connector for importing knowledge from files.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from knowledge.connectors.base_connector import BaseConnector
from knowledge.knowledge_base import KnowledgeBase
from knowledge.text_splitter import TextSplitter, get_text_splitter

logger = logging.getLogger(__name__)

class FileConnector(BaseConnector):
    """
    Connector for importing knowledge from files.
    
    This class provides methods for:
    - Loading text from files
    - Chunking text into smaller pieces
    - Adding the chunks to a knowledge base
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "simple",
    ):
        """
        Initialize the File Connector.
        
        Args:
            knowledge_base: Knowledge base to add the knowledge to
            chunk_size: Size of the chunks in characters
            chunk_overlap: Overlap between chunks in characters
            splitter_type: Type of text splitter to use
        """
        super().__init__(knowledge_base)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type
        
        # Initialize the text splitter
        self.text_splitter = get_text_splitter(
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    async def load_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Load a file and add its contents to the knowledge base.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata to add to the chunks
            
        Returns:
            List of IDs for the added chunks
        """
        # Convert string path to Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Ensure the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create base metadata
        if metadata is None:
            metadata = {}
        
        # Add file information to metadata
        file_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size,
            "created_at": file_path.stat().st_ctime,
            "modified_at": file_path.stat().st_mtime,
            **metadata,
        }
        
        logger.info(f"Loading file: {file_path}")
        
        try:
            # Read the file
            text = await self._read_file(file_path)
            
            # Split the text into chunks
            chunks = self.text_splitter.split_text(text)
            
            logger.info(f"Split file into {len(chunks)} chunks")
            
            # Create metadata for each chunk
            metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    **file_metadata,
                }
                metadatas.append(chunk_metadata)
            
            # Add the chunks to the knowledge base
            ids = await self.knowledge_base.add_knowledge(chunks, metadatas)
            
            logger.info(f"Added {len(ids)} chunks to knowledge base")
            
            return ids
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    async def load_directory(
        self,
        directory_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Load all files in a directory and add their contents to the knowledge base.
        
        Args:
            directory_path: Path to the directory
            metadata: Optional metadata to add to the chunks
            extensions: Optional list of file extensions to include
            recursive: Whether to recursively load files in subdirectories
            
        Returns:
            Dictionary mapping file paths to lists of IDs for the added chunks
        """
        # Convert string path to Path object
        if isinstance(directory_path, str):
            directory_path = Path(directory_path)
        
        # Ensure the directory exists
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Ensure the path is a directory
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        # Normalize extensions
        if extensions:
            extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]
        
        logger.info(f"Loading directory: {directory_path}")
        
        # Get all files in the directory
        files = []
        if recursive:
            for root, _, filenames in os.walk(directory_path):
                for filename in filenames:
                    file_path = Path(root) / filename
                    if extensions is None or file_path.suffix.lower() in extensions:
                        files.append(file_path)
        else:
            for file_path in directory_path.iterdir():
                if file_path.is_file():
                    if extensions is None or file_path.suffix.lower() in extensions:
                        files.append(file_path)
        
        logger.info(f"Found {len(files)} files to load")
        
        # Load each file
        results = {}
        for file_path in files:
            try:
                # Create file-specific metadata
                file_metadata = {
                    "directory": str(directory_path),
                    "relative_path": str(file_path.relative_to(directory_path)),
                }
                
                if metadata:
                    file_metadata.update(metadata)
                
                # Load the file
                ids = await self.load_file(file_path, file_metadata)
                
                # Store the results
                results[str(file_path)] = ids
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                results[str(file_path)] = []
        
        return results
    
    async def _read_file(self, file_path: Path) -> str:
        """
        Read a file and return its contents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File contents as a string
        """
        # Get the file extension
        extension = file_path.suffix.lower()
        
        # Read the file based on its extension
        if extension in [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".csv", ".log"]:
            # Text files
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif extension in [".pdf"]:
            # PDF files
            try:
                import pypdf
                
                with open(file_path, "rb") as f:
                    pdf = pypdf.PdfReader(f)
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n\n"
                    return text
            except ImportError:
                logger.warning("pypdf not installed. Cannot read PDF files.")
                raise ImportError("pypdf not installed. Cannot read PDF files.")
        elif extension in [".docx"]:
            # Word documents
            try:
                import docx
                
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                logger.warning("python-docx not installed. Cannot read DOCX files.")
                raise ImportError("python-docx not installed. Cannot read DOCX files.")
        else:
            # Unsupported file type
            logger.warning(f"Unsupported file type: {extension}")
            raise ValueError(f"Unsupported file type: {extension}")

# Factory function to create a file connector
def create_file_connector(
    knowledge_base: KnowledgeBase,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    splitter_type: str = "simple",
) -> FileConnector:
    """
    Create a file connector.
    
    Args:
        knowledge_base: Knowledge base to add the knowledge to
        chunk_size: Size of the chunks in characters
        chunk_overlap: Overlap between chunks in characters
        splitter_type: Type of text splitter to use
        
    Returns:
        File connector instance
    """
    return FileConnector(
        knowledge_base=knowledge_base,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_type=splitter_type,
    ) 