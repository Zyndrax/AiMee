"""
Configuration management for AiMee.
Handles loading and accessing configuration from environment variables and config files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

class SupabaseConfig(BaseModel):
    """Supabase configuration for vector database."""
    url: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    key: str = Field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))
    table_name: str = Field(default="aimee_vectors")

class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    organization: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_ORGANIZATION", None))
    default_model: str = Field(default="gpt-4o")

class AnthropicConfig(BaseModel):
    """Anthropic API configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    default_model: str = Field(default="claude-3-opus-20240229")

class AIModelConfig(BaseModel):
    """Configuration for AI models."""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    default_provider: str = Field(default="openai")

class PersonalityConfig(BaseModel):
    """Configuration for personality settings."""
    default_personality: str = Field(default="default")
    personalities_path: Path = Field(default=Path("personality/templates"))
    custom_personalities_path: Path = Field(default=Path("personality/custom"))

class AppConfig(BaseModel):
    """Main application configuration."""
    app_name: str = Field(default="AiMee")
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    supabase: SupabaseConfig = Field(default_factory=SupabaseConfig)
    ai_models: AIModelConfig = Field(default_factory=AIModelConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    
    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> "AppConfig":
        """Load configuration from a file."""
        # TODO: Implement loading from YAML/JSON config file
        return cls()

# Global configuration instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config 