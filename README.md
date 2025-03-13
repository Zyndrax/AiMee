# AiMee

AiMee is a modular AI-powered assistant platform designed for flexibility, extensibility, and cross-platform compatibility.

## Overview

AiMee is built with modularity at its core, allowing its components to be easily integrated into other projects or applications. The platform consists of multiple specialized AI agents that can work together or independently, with a pluggable architecture that enables easy extension and customization.

## Key Features

- **Modular Architecture**: Each component is designed to be used independently or as part of the larger system
- **Multiple AI Agents**: Specialized agents for different tasks that can be used across projects
- **Pluggable Personality Cores**: Customizable personality modules that users can modify or create
- **Vectorized Knowledge Base**: Supabase-powered vector database for efficient knowledge storage and retrieval
- **Model Selection Logic**: Intelligent routing to determine the optimal AI model for each task
- **Cross-Platform Compatibility**: Designed to work across various platforms and environments
- **MCP Integration**: Support for multiple MCPs (Model Control Protocols) for enhanced capabilities

## Project Structure

```
AiMee/
├── core/                  # Core platform functionality
│   ├── agent_manager/     # Manages and coordinates different AI agents
│   ├── model_router/      # Intelligent routing between different AI models
│   └── config/            # Configuration management
├── agents/                # Individual AI agents with specific capabilities
│   ├── conversation/      # Conversational agent
│   ├── research/          # Research and information gathering agent
│   ├── creative/          # Creative content generation agent
│   └── task/              # Task management and execution agent
├── personality/           # Personality core system
│   ├── core/              # Base personality framework
│   ├── templates/         # Pre-built personality templates
│   └── custom/            # Support for custom user-created personalities
├── knowledge/             # Knowledge management system
│   ├── vectordb/          # Supabase vector database integration
│   ├── connectors/        # External knowledge source connectors
│   └── indexing/          # Knowledge indexing and retrieval
├── platforms/             # Platform-specific implementations
│   ├── web/               # Web interface
│   ├── desktop/           # Desktop applications
│   ├── mobile/            # Mobile applications
│   └── api/               # API for third-party integration
├── mcp/                   # Model Control Protocol integrations
│   ├── openai/            # OpenAI MCP integration
│   ├── anthropic/         # Anthropic MCP integration
│   └── custom/            # Custom MCP implementations
└── utils/                 # Shared utilities and helpers
```

## Getting Started

More details coming soon.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 