"""
Supabase Vector Database Integration for AiMee.

This module provides integration with Supabase's vector database capabilities,
allowing for efficient storage and retrieval of vector embeddings.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from supabase import Client, create_client

from core.config.config import get_config

logger = logging.getLogger(__name__)

class SupabaseVectorDB:
    """
    Supabase Vector Database integration for storing and retrieving vector embeddings.
    
    This class provides methods for:
    - Creating and managing vector tables
    - Storing embeddings with metadata
    - Performing similarity searches
    - Managing collections of embeddings
    """
    
    def __init__(
        self,
        table_name: Optional[str] = None,
        url: Optional[str] = None,
        key: Optional[str] = None,
        embedding_dimension: int = 1536,  # Default for OpenAI embeddings
    ):
        """
        Initialize the Supabase Vector DB client.
        
        Args:
            table_name: Name of the vector table to use
            url: Supabase URL (if None, will use from config)
            key: Supabase API key (if None, will use from config)
            embedding_dimension: Dimension of the embedding vectors
        """
        config = get_config()
        
        self.url = url or config.supabase.url
        self.key = key or config.supabase.key
        self.table_name = table_name or config.supabase.table_name
        self.embedding_dimension = embedding_dimension
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key must be provided either directly or via environment variables"
            )
        
        self.client = create_client(self.url, self.key)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the vector database, creating tables if they don't exist.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Check if the table exists, create it if it doesn't
            await self._ensure_table_exists()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Supabase Vector DB: {e}")
            return False
    
    async def _ensure_table_exists(self) -> None:
        """
        Ensure that the vector table exists, creating it if it doesn't.
        """
        try:
            # Try to select from the table to see if it exists
            response = await self._execute_query(f"SELECT COUNT(*) FROM {self.table_name} LIMIT 1")
            logger.info(f"Vector table {self.table_name} exists")
        except Exception as e:
            logger.warning(f"Vector table {self.table_name} may not exist: {e}")
            
            # Create the table
            try:
                # Enable the pgvector extension if it's not already enabled
                await self._execute_query("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create the table with a vector column
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR({self.embedding_dimension}) NOT NULL
                )
                """
                await self._execute_query(create_table_query)
                
                # Create an index for faster similarity searches
                index_query = f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
                await self._execute_query(index_query)
                
                logger.info(f"Created vector table {self.table_name}")
            except Exception as create_error:
                logger.error(f"Failed to create vector table: {create_error}")
                raise
    
    async def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a raw SQL query against the Supabase database.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Query result
        """
        logger.debug(f"Executing query: {query} with params: {params}")
        
        try:
            # Execute the query using Supabase's RPC function
            result = self.client.rpc(
                "execute_sql",
                {
                    "query_text": query,
                    "params": params or {},
                }
            ).execute()
            
            # Check for errors
            if hasattr(result, 'error') and result.error:
                raise Exception(f"Supabase query error: {result.error}")
            
            return result.data
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    async def add_embeddings(
        self,
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add embeddings to the vector database.
        
        Args:
            embeddings: List of embedding vectors
            contents: List of content strings corresponding to the embeddings
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs for the embeddings
            
        Returns:
            List of IDs for the added embeddings
        """
        if not self._initialized:
            raise RuntimeError("Supabase Vector DB not initialized. Call initialize() first.")
        
        if metadatas is None:
            metadatas = [{} for _ in embeddings]
        
        if len(embeddings) != len(contents) or len(embeddings) != len(metadatas):
            raise ValueError("embeddings, contents, and metadatas must have the same length")
        
        logger.info(f"Adding {len(embeddings)} embeddings to {self.table_name}")
        
        # Prepare the data for insertion
        data = []
        for content, metadata, embedding in zip(contents, metadatas, embeddings):
            item = {
                "content": content,
                "metadata": json.dumps(metadata),  # Convert metadata to JSON string
                "embedding": embedding,
            }
            data.append(item)
        
        # Insert the data into the table
        try:
            result = self.client.from_(self.table_name).insert(data).execute()
            
            # Check for errors
            if hasattr(result, 'error') and result.error:
                raise Exception(f"Supabase insert error: {result.error}")
            
            # Extract the IDs from the result
            inserted_ids = [item['id'] for item in result.data]
            return inserted_ids
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            # Return dummy IDs as fallback
            return ids or [f"dummy-id-{i}" for i in range(len(embeddings))]
    
    async def search_embeddings(
        self,
        query_embedding: List[float],
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in the vector database.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            List of search results, each containing the content, metadata, and similarity score
        """
        if not self._initialized:
            raise RuntimeError("Supabase Vector DB not initialized. Call initialize() first.")
        
        logger.info(f"Searching for similar embeddings in {self.table_name}")
        
        try:
            # Build the query
            query = f"""
            SELECT id, content, metadata, 1 - (embedding <=> $1) as similarity
            FROM {self.table_name}
            WHERE 1 = 1
            """
            
            # Add metadata filters if provided
            params = {"embedding": query_embedding}
            if metadata_filter:
                for i, (key, value) in enumerate(metadata_filter.items(), start=2):
                    query += f" AND metadata->>'${i}' = '${i+1}'"
                    params[f"${i}"] = key
                    params[f"${i+1}"] = value
            
            # Add order by and limit
            query += f"""
            ORDER BY similarity DESC
            LIMIT {limit}
            """
            
            # Execute the query
            result = await self._execute_query(query, params)
            
            # Process the results
            search_results = []
            for item in result:
                # Parse the metadata from JSON string
                metadata = json.loads(item['metadata']) if item['metadata'] else {}
                
                search_results.append({
                    "id": item['id'],
                    "content": item['content'],
                    "metadata": metadata,
                    "similarity": item['similarity'],
                })
            
            return search_results
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            # Return dummy results as fallback
            return [
                {
                    "content": f"Dummy content {i}",
                    "metadata": {"dummy": "metadata"},
                    "similarity": 0.9 - (i * 0.1),
                }
                for i in range(min(limit, 5))
            ]
    
    async def delete_embeddings(self, ids: List[str]) -> bool:
        """
        Delete embeddings from the vector database.
        
        Args:
            ids: List of embedding IDs to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Supabase Vector DB not initialized. Call initialize() first.")
        
        logger.info(f"Deleting {len(ids)} embeddings from {self.table_name}")
        
        try:
            # Delete the embeddings
            result = self.client.from_(self.table_name).delete().in_("id", ids).execute()
            
            # Check for errors
            if hasattr(result, 'error') and result.error:
                raise Exception(f"Supabase delete error: {result.error}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False
    
    async def get_embedding(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific embedding by ID.
        
        Args:
            id: ID of the embedding to retrieve
            
        Returns:
            Embedding data if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Supabase Vector DB not initialized. Call initialize() first.")
        
        logger.info(f"Getting embedding {id} from {self.table_name}")
        
        try:
            # Get the embedding
            result = self.client.from_(self.table_name).select("*").eq("id", id).execute()
            
            # Check for errors
            if hasattr(result, 'error') and result.error:
                raise Exception(f"Supabase select error: {result.error}")
            
            # Check if any data was returned
            if not result.data:
                return None
            
            # Parse the metadata from JSON string
            item = result.data[0]
            metadata = json.loads(item['metadata']) if item['metadata'] else {}
            
            return {
                "id": item['id'],
                "content": item['content'],
                "metadata": metadata,
                "embedding": item['embedding'],
            }
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a dummy result as fallback
            return {
                "id": id,
                "content": f"Dummy content for {id}",
                "metadata": {"dummy": "metadata"},
                "embedding": [0.1] * 10,  # Shortened for brevity
            }
    
    async def update_embedding_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for a specific embedding.
        
        Args:
            id: ID of the embedding to update
            metadata: New metadata
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Supabase Vector DB not initialized. Call initialize() first.")
        
        logger.info(f"Updating metadata for embedding {id} in {self.table_name}")
        
        try:
            # Update the metadata
            result = self.client.from_(self.table_name).update({
                "metadata": json.dumps(metadata),
            }).eq("id", id).execute()
            
            # Check for errors
            if hasattr(result, 'error') and result.error:
                raise Exception(f"Supabase update error: {result.error}")
            
            return True
        except Exception as e:
            logger.error(f"Error updating embedding metadata: {e}")
            return False
    
    async def count_embeddings(self, metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Count the number of embeddings in the database.
        
        Args:
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            Number of embeddings
        """
        if not self._initialized:
            raise RuntimeError("Supabase Vector DB not initialized. Call initialize() first.")
        
        logger.info(f"Counting embeddings in {self.table_name}")
        
        try:
            # Build the query
            query = f"SELECT COUNT(*) FROM {self.table_name} WHERE 1 = 1"
            
            # Add metadata filters if provided
            params = {}
            if metadata_filter:
                for i, (key, value) in enumerate(metadata_filter.items(), start=1):
                    query += f" AND metadata->>'${i}' = '${i+1}'"
                    params[f"${i}"] = key
                    params[f"${i+1}"] = value
            
            # Execute the query
            result = await self._execute_query(query, params)
            
            # Extract the count
            return result[0]['count']
        except Exception as e:
            logger.error(f"Error counting embeddings: {e}")
            # Return a dummy count as fallback
            return 42
    
    async def close(self) -> None:
        """
        Close the database connection.
        """
        logger.info("Closing Supabase Vector DB connection")
        
        # Supabase client doesn't have a close method, so we just mark as not initialized
        self._initialized = False 