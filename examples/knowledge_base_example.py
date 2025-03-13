"""
Knowledge Base Example for AiMee.

This example demonstrates how to use the knowledge base and vector database
to store and retrieve information.
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from knowledge.connectors.file_connector import FileConnector
from knowledge.knowledge_base import KnowledgeBase
from mcp.embeddings_factory import get_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def main():
    """
    Main function to demonstrate the knowledge base.
    """
    logger.info("Starting knowledge base example")
    
    # Initialize the knowledge base
    kb = KnowledgeBase(
        namespace="example",
        vector_db_provider="supabase",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )
    
    # Initialize the knowledge base
    logger.info("Initializing knowledge base")
    success = await kb.initialize()
    
    if not success:
        logger.error("Failed to initialize knowledge base")
        return
    
    # Create a file connector
    logger.info("Creating file connector")
    file_connector = FileConnector(
        knowledge_base=kb,
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    # Create a sample text file
    sample_file = Path("sample.txt")
    with open(sample_file, "w") as f:
        f.write("""
        # AiMee: AI Memory and Knowledge Management System

        AiMee is an advanced AI memory and knowledge management system designed to enhance AI applications with persistent memory and knowledge retrieval capabilities.

        ## Features

        - **Vector Database Integration**: Store and retrieve information using semantic similarity.
        - **Knowledge Base Management**: Organize knowledge in namespaces and collections.
        - **Connector System**: Import knowledge from various sources like files, APIs, and databases.
        - **Embedding Models**: Support for multiple embedding models from providers like OpenAI and Anthropic.
        - **Async Support**: Built with asyncio for efficient concurrent operations.
        - **Extensible Architecture**: Easily add new vector databases, embedding models, and connectors.

        ## Use Cases

        - **Chatbots with Memory**: Create chatbots that remember conversation history and user preferences.
        - **Document Q&A**: Build systems that can answer questions about documents and knowledge bases.
        - **Semantic Search**: Implement semantic search for your applications.
        - **Knowledge Management**: Organize and retrieve knowledge from various sources.
        """)
    
    try:
        # Load the sample file
        logger.info("Loading sample file")
        ids = await file_connector.load_file(sample_file)
        logger.info(f"Added {len(ids)} chunks to knowledge base")
        
        # Search for knowledge
        logger.info("Searching for knowledge about vector database")
        results = await kb.search_knowledge("vector database integration")
        
        # Display the results
        logger.info(f"Found {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  Content: {result['text'][:100]}...")
            logger.info(f"  Score: {result['score']}")
            logger.info(f"  Metadata: {result['metadata']}")
        
        # Count knowledge
        count = await kb.count_knowledge()
        logger.info(f"Total chunks in knowledge base: {count}")
        
        # Clean up
        logger.info("Deleting knowledge")
        success = await kb.delete_knowledge(ids)
        logger.info(f"Deletion {'successful' if success else 'failed'}")
        
    finally:
        # Close the knowledge base
        logger.info("Closing knowledge base")
        await kb.close()
        
        # Remove the sample file
        if sample_file.exists():
            sample_file.unlink()
            logger.info("Removed sample file")

if __name__ == "__main__":
    asyncio.run(main()) 