import os
import sys

# Define the directory structure
directories = [
    # Core
    os.path.join("core", "agent_manager"),
    os.path.join("core", "model_router"),
    os.path.join("core", "config"),
    
    # Agents
    os.path.join("agents", "conversation"),
    os.path.join("agents", "research"),
    os.path.join("agents", "creative"),
    os.path.join("agents", "task"),
    
    # Personality
    os.path.join("personality", "core"),
    os.path.join("personality", "templates"),
    os.path.join("personality", "custom"),
    
    # Knowledge
    os.path.join("knowledge", "vectordb"),
    os.path.join("knowledge", "connectors"),
    os.path.join("knowledge", "indexing"),
    
    # Platforms
    os.path.join("platforms", "web"),
    os.path.join("platforms", "desktop"),
    os.path.join("platforms", "mobile"),
    os.path.join("platforms", "api"),
    
    # MCP
    os.path.join("mcp", "openai"),
    os.path.join("mcp", "anthropic"),
    os.path.join("mcp", "custom"),
    
    # Utils
    "utils"
]

# Create directories
for directory in directories:
    try:
        path = os.path.join(os.getcwd(), directory)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    except Exception as e:
        print(f"Error creating {directory}: {e}", file=sys.stderr)

print("Directory structure creation completed!") 