"""
Example script demonstrating how to use the AiMee knowledge base with Supabase vector database.

This script shows how to:
1. Initialize the knowledge base
2. Add knowledge to the knowledge base
3. Search for knowledge in the knowledge base
4. Import knowledge from files using the file connector
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from knowledge.connectors.file_connector import create_file_connector
from knowledge.knowledge_base import get_knowledge_base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

async def main():
    """Run the knowledge base example."""
    logger.info("Starting knowledge base example")
    
    # Check if Supabase credentials are set
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        logger.warning(
            "Supabase credentials not set. This example will use dummy implementations. "
            "Set SUPABASE_URL and SUPABASE_KEY environment variables for real functionality."
        )
    
    # Initialize the knowledge base
    kb = get_knowledge_base(namespace="example")
    await kb.initialize()
    logger.info("Knowledge base initialized")
    
    # Add some knowledge directly
    knowledge_ids = await kb.add_knowledge(
        texts=[
            "AiMee is a modular AI-powered assistant platform.",
            "AiMee is designed for flexibility, extensibility, and cross-platform compatibility.",
            "AiMee has a pluggable architecture that enables easy extension and customization.",
        ],
        metadata=[
            {"category": "overview", "importance": "high"},
            {"category": "design", "importance": "medium"},
            {"category": "architecture", "importance": "high"},
        ],
    )
    logger.info(f"Added {len(knowledge_ids)} knowledge items directly")
    
    # Search for knowledge
    results = await kb.search_knowledge(query="modular architecture", limit=5)
    logger.info(f"Found {len(results)} results for 'modular architecture'")
    
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}:")
        logger.info(f"  Content: {result['content']}")
        logger.info(f"  Metadata: {result['metadata']}")
        logger.info(f"  Similarity: {result['similarity']}")
    
    # Import knowledge from files
    # Create a file connector
    examples_dir = Path(__file__).parent
    file_connector = create_file_connector(
        namespace="example_files",
        base_path=examples_dir,
        file_extensions=[".txt", ".md"],
    )
    
    # Initialize the connector
    await file_connector.initialize()
    logger.info("File connector initialized")
    
    # Create a sample file for demonstration
    sample_file = examples_dir / "sample_knowledge.md"
    with open(sample_file, "w") as f:
        f.write("""# AiMee Knowledge Base Example

This is a sample file for demonstrating the knowledge base functionality.

## Features

- Vectorized knowledge storage
- Semantic search capabilities
- Modular architecture
- Cross-platform compatibility

## Use Cases

1. Storing and retrieving information
2. Building a knowledge graph
3. Enhancing AI capabilities with domain-specific knowledge
""")
    
    # Import knowledge from the sample file
    file_ids = await file_connector.import_knowledge(path="sample_knowledge.md")
    logger.info(f"Imported {len(file_ids)} knowledge items from files")
    
    # Search for knowledge in the file namespace
    file_kb = get_knowledge_base(namespace="example_files")
    file_results = await file_kb.search_knowledge(query="semantic search", limit=3)
    logger.info(f"Found {len(file_results)} results for 'semantic search' in files")
    
    for i, result in enumerate(file_results):
        logger.info(f"File Result {i+1}:")
        logger.info(f"  Content: {result['content'][:100]}...")
        logger.info(f"  Metadata: {result['metadata']}")
        logger.info(f"  Similarity: {result['similarity']}")
    
    # Clean up
    await kb.close()
    await file_connector.close()
    logger.info("Knowledge base example completed")

if __name__ == "__main__":
    asyncio.run(main()) 