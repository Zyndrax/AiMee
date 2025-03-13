"""
Agent Manager for AiMee.
Handles the registration, initialization, and coordination of agents.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type, Union

from .base_agent import BaseAgent

class AgentManager:
    """
    Manages the lifecycle and coordination of agents in the AiMee system.
    
    The AgentManager is responsible for:
    - Registering agents
    - Initializing agents
    - Routing requests to appropriate agents
    - Coordinating communication between agents
    - Shutting down agents
    """
    
    def __init__(self):
        """Initialize the agent manager."""
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_types: Dict[str, Type[BaseAgent]] = {}
    
    def register_agent_type(self, agent_type: Type[BaseAgent], type_name: Optional[str] = None) -> None:
        """
        Register an agent type that can be instantiated later.
        
        Args:
            agent_type: The agent class to register
            type_name: Optional name for the agent type. If not provided, the class name is used.
        """
        name = type_name or agent_type.__name__
        self._agent_types[name] = agent_type
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent instance with the manager.
        
        Args:
            agent: The agent instance to register
        
        Raises:
            ValueError: If an agent with the same name is already registered
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent with name '{agent.name}' is already registered")
        
        self._agents[agent.name] = agent
    
    def create_agent(self, agent_type: str, name: str, **kwargs) -> BaseAgent:
        """
        Create and register a new agent of the specified type.
        
        Args:
            agent_type: The type of agent to create
            name: The name for the new agent
            **kwargs: Additional arguments to pass to the agent constructor
        
        Returns:
            The created agent instance
            
        Raises:
            ValueError: If the agent type is not registered or an agent with the same name exists
        """
        if agent_type not in self._agent_types:
            raise ValueError(f"Agent type '{agent_type}' is not registered")
        
        if name in self._agents:
            raise ValueError(f"Agent with name '{name}' is already registered")
        
        agent_class = self._agent_types[agent_type]
        agent = agent_class(name=name, **kwargs)
        self.register_agent(agent)
        return agent
    
    def get_agent(self, name: str) -> BaseAgent:
        """
        Get an agent by name.
        
        Args:
            name: The name of the agent to retrieve
            
        Returns:
            The agent instance
            
        Raises:
            KeyError: If no agent with the given name is registered
        """
        if name not in self._agents:
            raise KeyError(f"No agent registered with name '{name}'")
        
        return self._agents[name]
    
    def list_agents(self) -> List[str]:
        """
        Get a list of all registered agent names.
        
        Returns:
            List[str]: A list of agent names
        """
        return list(self._agents.keys())
    
    def list_agent_types(self) -> List[str]:
        """
        Get a list of all registered agent types.
        
        Returns:
            List[str]: A list of agent type names
        """
        return list(self._agent_types.keys())
    
    async def initialize_agent(self, name: str) -> bool:
        """
        Initialize a specific agent.
        
        Args:
            name: The name of the agent to initialize
            
        Returns:
            bool: True if initialization was successful, False otherwise
            
        Raises:
            KeyError: If no agent with the given name is registered
        """
        agent = self.get_agent(name)
        return await agent.initialize()
    
    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered agents.
        
        Returns:
            Dict[str, bool]: A dictionary mapping agent names to initialization success status
        """
        results = {}
        for name, agent in self._agents.items():
            results[name] = await agent.initialize()
        return results
    
    async def process(self, agent_name: str, input_data: Any) -> Any:
        """
        Process input data with a specific agent.
        
        Args:
            agent_name: The name of the agent to use
            input_data: The input data to process
            
        Returns:
            The processed result
            
        Raises:
            KeyError: If no agent with the given name is registered
            RuntimeError: If the agent is not initialized
        """
        agent = self.get_agent(agent_name)
        
        if not agent.is_initialized:
            raise RuntimeError(f"Agent '{agent_name}' is not initialized")
        
        return await agent.process(input_data)
    
    async def shutdown_agent(self, name: str) -> bool:
        """
        Shut down a specific agent.
        
        Args:
            name: The name of the agent to shut down
            
        Returns:
            bool: True if shutdown was successful, False otherwise
            
        Raises:
            KeyError: If no agent with the given name is registered
        """
        agent = self.get_agent(name)
        return await agent.shutdown()
    
    async def shutdown_all(self) -> Dict[str, bool]:
        """
        Shut down all registered agents.
        
        Returns:
            Dict[str, bool]: A dictionary mapping agent names to shutdown success status
        """
        results = {}
        for name, agent in self._agents.items():
            results[name] = await agent.shutdown()
        return results
    
    def remove_agent(self, name: str) -> None:
        """
        Remove an agent from the manager.
        
        Args:
            name: The name of the agent to remove
            
        Raises:
            KeyError: If no agent with the given name is registered
        """
        if name not in self._agents:
            raise KeyError(f"No agent registered with name '{name}'")
        
        del self._agents[name]

# Global agent manager instance
agent_manager = AgentManager()

def get_agent_manager() -> AgentManager:
    """Get the global agent manager instance."""
    return agent_manager 