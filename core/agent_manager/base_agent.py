"""
Base Agent class for AiMee.
Defines the interface that all agents must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class BaseAgent(ABC):
    """
    Base class for all AiMee agents.
    
    All agents in the system must inherit from this class and implement
    its abstract methods. This ensures a consistent interface across
    different agent implementations.
    """
    
    def __init__(self, name: str, description: str = "", config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new agent.
        
        Args:
            name: The name of the agent
            description: A description of the agent's purpose and capabilities
            config: Optional configuration parameters for the agent
        """
        self.name = name
        self.description = description
        self.config = config or {}
        self._is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent with any necessary setup.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Process input data and return a response.
        
        This is the main method that handles the agent's core functionality.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The processed result
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Perform any necessary cleanup when shutting down the agent.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the agent has been initialized."""
        return self._is_initialized
    
    @property
    def capabilities(self) -> List[str]:
        """
        Get a list of the agent's capabilities.
        
        Returns:
            List[str]: A list of capability identifiers
        """
        return []
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the agent.
        
        Returns:
            Dict[str, Any]: A dictionary containing agent information
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "is_initialized": self.is_initialized
        }
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self.is_initialized})" 