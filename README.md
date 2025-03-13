# AiMee: AI Memory and Knowledge Management System

AiMee is an advanced AI memory and knowledge management system designed to enhance AI applications with persistent memory and knowledge retrieval capabilities.

## Features

- **Vector Database Integration**: Store and retrieve information using semantic similarity with Supabase.
- **Knowledge Base Management**: Organize knowledge in namespaces and collections.
- **Connector System**: Import knowledge from various sources like files, APIs, and databases.
- **Embedding Models**: Support for multiple embedding models from providers like Deepseek, Replicate, OpenAI, and Anthropic.
- **Async Support**: Built with asyncio for efficient concurrent operations.
- **Extensible Architecture**: Easily add new vector databases, embedding models, and connectors.

## Setup

### Prerequisites

- Python 3.9+
- Supabase account
- Deepseek API key (or Replicate, OpenAI, Anthropic)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Zyndrax/AiMee.git
   cd AiMee
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file with your API keys and configuration.

### Setting up Supabase

AiMee uses Supabase with the pgvector extension for vector storage. You can set up the required tables automatically:

```bash
python scripts/setup_supabase.py
```

This script will:
- Enable the pgvector extension
- Create the embeddings table
- Set up the necessary indexes for similarity search

## Using AiMee with Deepseek

AiMee is configured to use Deepseek embeddings by default. Make sure you have a Deepseek API key and add it to your `.env` file:

```
DEEPSEEK_API_KEY=your-deepseek-api-key
EMBEDDING_PROVIDER=deepseek
EMBEDDING_MODEL=deepseek-ai/deepseek-embedding-v1
```

## Using AiMee with Replicate

To use Replicate for embeddings, update your `.env` file:

```
REPLICATE_API_KEY=your-replicate-api-key
EMBEDDING_PROVIDER=replicate
EMBEDDING_MODEL=owner/model:version
```

## Example Usage

Run the example script to see AiMee in action:

```bash
python examples/knowledge_base_example.py
```

This will:
1. Set up the Supabase tables
2. Initialize the knowledge base
3. Create a file connector
4. Load a sample text file
5. Perform a semantic search
6. Clean up the data

## Code Example

```python
import asyncio
from knowledge.knowledge_base import KnowledgeBase
from knowledge.connectors.file_connector import FileConnector

async def main():
    # Initialize the knowledge base
    kb = KnowledgeBase(
        namespace="my_namespace",
        vector_db_provider="supabase",
        embedding_provider="deepseek",
        embedding_model="deepseek-ai/deepseek-embedding-v1",
    )
    
    # Initialize the knowledge base
    await kb.initialize()
    
    # Create a file connector
    file_connector = FileConnector(
        knowledge_base=kb,
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    # Load a file
    ids = await file_connector.load_file("path/to/file.txt")
    
    # Search for knowledge
    results = await kb.search_knowledge("your search query")
    
    # Process results
    for result in results:
        print(f"Content: {result['text']}")
        print(f"Score: {result['score']}")
        print(f"Metadata: {result['metadata']}")
    
    # Clean up
    await kb.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Adding New Embedding Providers

AiMee is designed to be extensible. To add a new embedding provider:

1. Create a new file in the `mcp/` directory (e.g., `mcp/new_provider/embeddings.py`)
2. Implement the embedding class with `embed_documents` and `embed_query` methods
3. Update the `EmbeddingsFactory` in `mcp/embeddings_factory.py` to include your new provider

## License

MIT 