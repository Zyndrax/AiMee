"""
Text Splitter for AiMee.

This module provides text splitting functionality for breaking large texts
into smaller chunks for processing.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Type

logger = logging.getLogger(__name__)

class TextSplitter(ABC):
    """
    Abstract base class for text splitters.
    
    Text splitters are used to break large texts into smaller chunks
    for processing by embedding models and storage in vector databases.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        pass


class SimpleTextSplitter(TextSplitter):
    """
    Simple text splitter that splits text by paragraphs, sentences, or characters.
    
    This splitter tries to keep paragraphs and sentences intact when possible,
    falling back to character-level splitting when necessary.
    """
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # If text is shorter than chunk size, return it as is
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split by paragraphs first
        paragraphs = re.split(r"\n\s*\n", text)
        
        # If any paragraph is longer than chunk size, split it by sentences
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > self.chunk_size:
                # Split paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                
                # Replace the paragraph with its sentences
                paragraphs.pop(i)
                for j, sentence in enumerate(sentences):
                    paragraphs.insert(i + j, sentence)
        
        # Combine paragraphs into chunks
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph is too long, split it by characters
            if len(paragraph) > self.chunk_size:
                # If we have content in the current chunk, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split the paragraph by characters
                for i in range(0, len(paragraph), self.chunk_size - self.chunk_overlap):
                    chunk = paragraph[i:i + self.chunk_size]
                    chunks.append(chunk)
            
            # If adding the paragraph would exceed chunk size, start a new chunk
            elif len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = paragraph
            
            # Otherwise, add the paragraph to the current chunk
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class RecursiveTextSplitter(TextSplitter):
    """
    Recursive text splitter that splits text by a hierarchy of separators.
    
    This splitter tries to keep larger structures intact when possible,
    recursively falling back to smaller separators when necessary.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the recursive text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators to use, in order of preference
        """
        super().__init__(chunk_size, chunk_overlap)
        
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            "",      # Characters
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # If text is shorter than chunk size, return it as is
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator in turn
        for separator in self.separators:
            if separator == "":
                # Character-level splitting
                return [
                    text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
                ]
            
            # Split by the current separator
            splits = text.split(separator)
            
            # If we only get one split, continue to the next separator
            if len(splits) == 1:
                continue
            
            # Process each split recursively
            chunks = []
            current_chunk = []
            current_length = 0
            
            for split in splits:
                # Add the separator back to the split, except for the first one
                if current_chunk:
                    split_with_separator = separator + split
                else:
                    split_with_separator = split
                
                # If adding this split would exceed the chunk size, process the current chunk
                if current_length + len(split_with_separator) > self.chunk_size:
                    # If the current chunk is empty but the split is too long, recursively split it
                    if not current_chunk and len(split_with_separator) > self.chunk_size:
                        # Find the next separator in the list
                        next_separator_index = self.separators.index(separator) + 1
                        
                        # If we have more separators, recursively split with the next one
                        if next_separator_index < len(self.separators):
                            sub_splitter = RecursiveTextSplitter(
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.chunk_overlap,
                                separators=self.separators[next_separator_index:],
                            )
                            sub_chunks = sub_splitter.split_text(split_with_separator)
                            chunks.extend(sub_chunks)
                        else:
                            # No more separators, just add the split as is
                            chunks.append(split_with_separator)
                    else:
                        # Add the current chunk to the list of chunks
                        chunks.append(separator.join(current_chunk))
                        
                        # Start a new chunk with the current split
                        current_chunk = [split]
                        current_length = len(split)
                else:
                    # Add the split to the current chunk
                    current_chunk.append(split)
                    current_length += len(split_with_separator)
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(separator.join(current_chunk))
            
            # If we successfully split the text, return the chunks
            if chunks:
                return chunks
        
        # If we get here, we couldn't split the text
        return [text]


def get_text_splitter(
    splitter_type: str = "simple",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> TextSplitter:
    """
    Get a text splitter instance.
    
    Args:
        splitter_type: Type of text splitter to use
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Text splitter instance
    """
    splitter_map = {
        "simple": SimpleTextSplitter,
        "recursive": RecursiveTextSplitter,
    }
    
    splitter_class = splitter_map.get(splitter_type.lower())
    
    if not splitter_class:
        logger.warning(f"Unknown splitter type: {splitter_type}. Using SimpleTextSplitter.")
        splitter_class = SimpleTextSplitter
    
    return splitter_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ) 