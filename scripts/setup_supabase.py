"""
Supabase Setup Script for AiMee.

This script sets up the necessary tables and extensions in Supabase for AiMee.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional

import httpx
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SupabaseSetup:
    """
    Supabase setup class for AiMee.
    
    This class provides methods for setting up the necessary tables and extensions
    in Supabase for AiMee.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        table_name: Optional[str] = None,
        embedding_dimension: int = 1536,
    ):
        """
        Initialize the Supabase setup.
        
        Args:
            url: Supabase URL (if None, will use from environment)
            key: Supabase API key (if None, will use from environment)
            table_name: Name of the vector table to create (if None, will use from environment)
            embedding_dimension: Dimension of the embedding vectors
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        self.table_name = table_name or os.getenv("SUPABASE_TABLE_NAME", "embeddings")
        self.embedding_dimension = embedding_dimension
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key must be provided either directly or via environment variables"
            )
        
        self.client = create_client(self.url, self.key)
    
    async def setup(self) -> bool:
        """
        Set up the necessary tables and extensions in Supabase.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Enable the pgvector extension
            logger.info("Enabling pgvector extension...")
            await self._execute_query("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Enable the uuid-ossp extension for UUID generation
            logger.info("Enabling uuid-ossp extension...")
            await self._execute_query("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
            
            # Create the vector table
            logger.info(f"Creating vector table {self.table_name}...")
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
            logger.info(f"Creating index on {self.table_name}...")
            index_query = f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
            ON {self.table_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
            await self._execute_query(index_query)
            
            # Create a function to execute SQL queries
            logger.info("Creating execute_sql function...")
            execute_sql_function = """
            CREATE OR REPLACE FUNCTION execute_sql(query_text TEXT, params JSONB DEFAULT '{}'::jsonb)
            RETURNS JSONB
            LANGUAGE plpgsql
            SECURITY DEFINER
            AS $$
            DECLARE
                result JSONB;
            BEGIN
                EXECUTE query_text INTO result USING params;
                RETURN result;
            EXCEPTION WHEN OTHERS THEN
                RETURN jsonb_build_object(
                    'error', SQLERRM,
                    'detail', SQLSTATE
                );
            END;
            $$;
            """
            await self._execute_query(execute_sql_function)
            
            logger.info("Supabase setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up Supabase: {e}")
            return False
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a raw SQL query against the Supabase database.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query result
        """
        logger.debug(f"Executing query: {query}")
        
        try:
            # Execute the query using Supabase's REST API
            response = self.client.rpc("execute_sql", {"query_text": query}).execute()
            
            # Check for errors
            if hasattr(response, 'error') and response.error:
                # If the error is that the function doesn't exist, we need to create it first
                if "function execute_sql" in str(response.error) and "does not exist" in str(response.error):
                    logger.warning("execute_sql function doesn't exist yet, creating it directly...")
                    
                    # Create the function directly using the REST API
                    headers = {
                        "apikey": self.key,
                        "Authorization": f"Bearer {self.key}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    }
                    
                    sql_query = """
                    CREATE OR REPLACE FUNCTION execute_sql(query_text TEXT, params JSONB DEFAULT '{}'::jsonb)
                    RETURNS JSONB
                    LANGUAGE plpgsql
                    SECURITY DEFINER
                    AS $$
                    DECLARE
                        result JSONB;
                    BEGIN
                        EXECUTE query_text INTO result USING params;
                        RETURN result;
                    EXCEPTION WHEN OTHERS THEN
                        RETURN jsonb_build_object(
                            'error', SQLERRM,
                            'detail', SQLSTATE
                        );
                    END;
                    $$;
                    """
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{self.url}/rest/v1/rpc/execute_sql",
                            headers=headers,
                            json={"query_text": sql_query},
                        )
                        
                        if response.status_code != 200:
                            logger.error(f"Error creating execute_sql function: {response.text}")
                            raise Exception(f"Error creating execute_sql function: {response.text}")
                    
                    # Try executing the original query again
                    response = self.client.rpc("execute_sql", {"query_text": query}).execute()
                    
                    if hasattr(response, 'error') and response.error:
                        raise Exception(f"Supabase query error: {response.error}")
                else:
                    raise Exception(f"Supabase query error: {response.error}")
            
            return response.data
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

async def main():
    """
    Main function to set up Supabase.
    """
    logger.info("Starting Supabase setup")
    
    # Get the embedding dimension from the environment
    embedding_dimension = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    
    # Create the Supabase setup
    setup = SupabaseSetup(embedding_dimension=embedding_dimension)
    
    # Run the setup
    success = await setup.setup()
    
    if success:
        logger.info("Supabase setup completed successfully")
    else:
        logger.error("Supabase setup failed")

if __name__ == "__main__":
    asyncio.run(main()) 