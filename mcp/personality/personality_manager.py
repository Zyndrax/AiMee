"""
Personality Manager for AiMee.

This module provides a manager for different AI personalities, each with their own
knowledge bases, conversation history, and mood tracking.
"""

import os
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from knowledge.knowledge_base import KnowledgeBase
from knowledge.vectordb.supabase_vector_db import SupabaseVectorDB
from mcp.embeddings_factory import get_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Mood:
    """
    Represents the mood of a personality.
    
    Moods can influence how a personality responds to queries and interacts with users.
    """
    
    def __init__(
        self,
        name: str,
        intensity: float = 0.5,
        description: str = "",
        created_at: Optional[datetime] = None
    ):
        """
        Initialize a mood.
        
        Args:
            name: The name of the mood (e.g., "happy", "sad", "excited")
            intensity: The intensity of the mood, from 0.0 to 1.0
            description: A description of the mood
            created_at: When the mood was created/set
        """
        self.name = name
        self.intensity = max(0.0, min(1.0, intensity))  # Clamp between 0 and 1
        self.description = description
        self.created_at = created_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the mood to a dictionary for storage."""
        return {
            "name": self.name,
            "intensity": self.intensity,
            "description": self.description,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mood':
        """Create a mood from a dictionary."""
        return cls(
            name=data["name"],
            intensity=data["intensity"],
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"])
        )


class Memory:
    """
    Represents a memory for a personality.
    
    Memories can be core traits, experiences, or knowledge that define a personality.
    """
    
    def __init__(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a memory.
        
        Args:
            content: The content of the memory
            memory_type: The type of memory (e.g., "core_trait", "experience", "knowledge")
            importance: How important this memory is (0.0 to 1.0)
            created_at: When the memory was created
            metadata: Additional metadata about the memory
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.importance = max(0.0, min(1.0, importance))  # Clamp between 0 and 1
        self.created_at = created_at or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory to a dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create a memory from a dictionary."""
        memory = cls(
            content=data["content"],
            memory_type=data["memory_type"],
            importance=data["importance"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data["metadata"]
        )
        memory.id = data["id"]
        return memory


class Personality:
    """
    Represents an AI personality with its own knowledge base, mood, and memories.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        knowledge_base: Optional[KnowledgeBase] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize a personality.
        
        Args:
            name: The name of the personality
            description: A description of the personality
            knowledge_base: An existing knowledge base to use (if None, one will be created)
            embedding_provider: The embedding provider to use (if None, will use from config)
            embedding_model: The embedding model to use (if None, will use from config)
        """
        self.name = name
        self.description = description
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.current_mood: Optional[Mood] = None
        self.mood_history: List[Mood] = []
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Set up the knowledge base
        if knowledge_base:
            self.knowledge_base = knowledge_base
        else:
            # Create a new knowledge base with a Supabase vector DB
            vector_db = SupabaseVectorDB(
                collection_name=f"personality_{name.lower().replace(' ', '_')}",
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            )
            self.knowledge_base = KnowledgeBase(
                vector_db=vector_db,
                name=f"{name} Knowledge Base"
            )
    
    def add_memory(self, memory: Memory) -> str:
        """
        Add a memory to the personality's knowledge base.
        
        Args:
            memory: The memory to add
            
        Returns:
            The ID of the added memory
        """
        # Add the memory to the knowledge base
        metadata = memory.to_dict()
        self.knowledge_base.add_knowledge(
            content=memory.content,
            metadata=metadata
        )
        
        logger.info(f"Added memory to {self.name}: {memory.content[:50]}...")
        return memory.id
    
    def retrieve_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 5
    ) -> List[Memory]:
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: The query to search for
            memory_type: Filter by memory type (if provided)
            min_importance: Minimum importance threshold
            limit: Maximum number of memories to retrieve
            
        Returns:
            A list of relevant memories
        """
        # Search the knowledge base
        results = self.knowledge_base.search_knowledge(
            query=query,
            limit=limit
        )
        
        # Convert results to Memory objects and filter
        memories = []
        for result in results:
            try:
                memory = Memory.from_dict(result.metadata)
                
                # Apply filters
                if memory_type and memory.memory_type != memory_type:
                    continue
                    
                if memory.importance < min_importance:
                    continue
                    
                memories.append(memory)
            except Exception as e:
                logger.error(f"Error converting result to memory: {e}")
        
        return memories
    
    def set_mood(self, mood: Mood) -> None:
        """
        Set the current mood of the personality.
        
        Args:
            mood: The new mood
        """
        # Add the current mood to history if it exists
        if self.current_mood:
            self.mood_history.append(self.current_mood)
        
        # Set the new mood
        self.current_mood = mood
        logger.info(f"Set mood for {self.name}: {mood.name} (intensity: {mood.intensity})")
    
    def add_conversation(self, user_message: str, ai_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a conversation exchange to the personality's history.
        
        Args:
            user_message: The message from the user
            ai_response: The response from the AI
            metadata: Additional metadata about the conversation
        """
        conversation = {
            "user_message": user_message,
            "ai_response": ai_response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to conversation history
        self.conversation_history.append(conversation)
        
        # Optionally, add important conversations to the knowledge base
        # This could be based on some importance criteria
        if metadata and metadata.get("important", False):
            memory = Memory(
                content=f"User: {user_message}\nAI: {ai_response}",
                memory_type="conversation",
                importance=metadata.get("importance", 0.5),
                metadata=metadata
            )
            self.add_memory(memory)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the personality to a dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "current_mood": self.current_mood.to_dict() if self.current_mood else None,
            "mood_history": [mood.to_dict() for mood in self.mood_history],
            "conversation_history": self.conversation_history
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        knowledge_base: Optional[KnowledgeBase] = None
    ) -> 'Personality':
        """Create a personality from a dictionary."""
        personality = cls(
            name=data["name"],
            description=data["description"],
            knowledge_base=knowledge_base
        )
        
        personality.id = data["id"]
        personality.created_at = datetime.fromisoformat(data["created_at"])
        
        if data["current_mood"]:
            personality.current_mood = Mood.from_dict(data["current_mood"])
            
        personality.mood_history = [Mood.from_dict(mood) for mood in data["mood_history"]]
        personality.conversation_history = data["conversation_history"]
        
        return personality


class PersonalityManager:
    """
    Manages multiple AI personalities.
    
    This class provides methods for creating, retrieving, and managing different
    AI personalities, each with their own knowledge bases, moods, and memories.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the personality manager.
        
        Args:
            embedding_provider: The embedding provider to use (if None, will use from config)
            embedding_model: The embedding model to use (if None, will use from config)
        """
        self.personalities: Dict[str, Personality] = {}
        self.active_personality_id: Optional[str] = None
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        
        # Initialize the embedding model
        self.embeddings = get_embeddings(
            provider=embedding_provider,
            model=embedding_model
        )
        
        logger.info(f"Initialized PersonalityManager with {embedding_provider or 'default'} embeddings")
    
    def create_personality(
        self,
        name: str,
        description: str = "",
        initial_memories: Optional[List[Memory]] = None,
        initial_mood: Optional[Mood] = None
    ) -> Personality:
        """
        Create a new personality.
        
        Args:
            name: The name of the personality
            description: A description of the personality
            initial_memories: Initial memories to add to the personality
            initial_mood: Initial mood for the personality
            
        Returns:
            The created personality
        """
        # Create the personality
        personality = Personality(
            name=name,
            description=description,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model
        )
        
        # Add initial memories
        if initial_memories:
            for memory in initial_memories:
                personality.add_memory(memory)
        
        # Set initial mood
        if initial_mood:
            personality.set_mood(initial_mood)
        
        # Store the personality
        self.personalities[personality.id] = personality
        
        # If this is the first personality, make it active
        if not self.active_personality_id:
            self.active_personality_id = personality.id
        
        logger.info(f"Created personality: {name} (ID: {personality.id})")
        return personality
    
    def get_personality(self, personality_id: str) -> Optional[Personality]:
        """
        Get a personality by ID.
        
        Args:
            personality_id: The ID of the personality
            
        Returns:
            The personality, or None if not found
        """
        return self.personalities.get(personality_id)
    
    def get_personality_by_name(self, name: str) -> Optional[Personality]:
        """
        Get a personality by name.
        
        Args:
            name: The name of the personality
            
        Returns:
            The personality, or None if not found
        """
        for personality in self.personalities.values():
            if personality.name.lower() == name.lower():
                return personality
        return None
    
    def set_active_personality(self, personality_id: str) -> bool:
        """
        Set the active personality.
        
        Args:
            personality_id: The ID of the personality to set as active
            
        Returns:
            True if successful, False if the personality was not found
        """
        if personality_id in self.personalities:
            self.active_personality_id = personality_id
            logger.info(f"Set active personality: {self.personalities[personality_id].name}")
            return True
        return False
    
    def get_active_personality(self) -> Optional[Personality]:
        """
        Get the currently active personality.
        
        Returns:
            The active personality, or None if no personality is active
        """
        if self.active_personality_id:
            return self.personalities.get(self.active_personality_id)
        return None
    
    def list_personalities(self) -> List[Dict[str, str]]:
        """
        List all personalities.
        
        Returns:
            A list of dictionaries with personality ID, name, and description
        """
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description
            }
            for p in self.personalities.values()
        ]
    
    def delete_personality(self, personality_id: str) -> bool:
        """
        Delete a personality.
        
        Args:
            personality_id: The ID of the personality to delete
            
        Returns:
            True if successful, False if the personality was not found
        """
        if personality_id in self.personalities:
            personality = self.personalities[personality_id]
            
            # Clean up the knowledge base
            try:
                personality.knowledge_base.close()
            except Exception as e:
                logger.error(f"Error closing knowledge base: {e}")
            
            # Remove the personality
            del self.personalities[personality_id]
            
            # If this was the active personality, clear the active personality
            if self.active_personality_id == personality_id:
                self.active_personality_id = None
                
                # Set a new active personality if there are any left
                if self.personalities:
                    self.active_personality_id = next(iter(self.personalities.keys()))
            
            logger.info(f"Deleted personality: {personality.name}")
            return True
        
        return False
    
    def add_memory_to_active(self, memory: Memory) -> Optional[str]:
        """
        Add a memory to the active personality.
        
        Args:
            memory: The memory to add
            
        Returns:
            The ID of the added memory, or None if no active personality
        """
        personality = self.get_active_personality()
        if personality:
            return personality.add_memory(memory)
        return None
    
    def set_mood_for_active(self, mood: Mood) -> bool:
        """
        Set the mood for the active personality.
        
        Args:
            mood: The mood to set
            
        Returns:
            True if successful, False if no active personality
        """
        personality = self.get_active_personality()
        if personality:
            personality.set_mood(mood)
            return True
        return False
    
    def retrieve_memories_from_active(
        self,
        query: str,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 5
    ) -> List[Memory]:
        """
        Retrieve memories from the active personality.
        
        Args:
            query: The query to search for
            memory_type: Filter by memory type (if provided)
            min_importance: Minimum importance threshold
            limit: Maximum number of memories to retrieve
            
        Returns:
            A list of relevant memories, or empty list if no active personality
        """
        personality = self.get_active_personality()
        if personality:
            return personality.retrieve_memories(
                query=query,
                memory_type=memory_type,
                min_importance=min_importance,
                limit=limit
            )
        return []
    
    def add_conversation_to_active(
        self,
        user_message: str,
        ai_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a conversation to the active personality.
        
        Args:
            user_message: The message from the user
            ai_response: The response from the AI
            metadata: Additional metadata about the conversation
            
        Returns:
            True if successful, False if no active personality
        """
        personality = self.get_active_personality()
        if personality:
            personality.add_conversation(
                user_message=user_message,
                ai_response=ai_response,
                metadata=metadata
            )
            return True
        return False
    
    def save_personalities(self, file_path: str) -> bool:
        """
        Save all personalities to a file.
        
        Args:
            file_path: The path to save the personalities to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            # Convert personalities to dictionaries
            personalities_dict = {
                p_id: p.to_dict()
                for p_id, p in self.personalities.items()
            }
            
            # Add active personality ID
            data = {
                "active_personality_id": self.active_personality_id,
                "personalities": personalities_dict
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.personalities)} personalities to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving personalities: {e}")
            return False
    
    def load_personalities(self, file_path: str) -> bool:
        """
        Load personalities from a file.
        
        Args:
            file_path: The path to load the personalities from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            # Load from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Clear existing personalities
            self.personalities = {}
            
            # Load personalities
            for p_id, p_data in data["personalities"].items():
                # Create a vector DB for the personality
                vector_db = SupabaseVectorDB(
                    collection_name=f"personality_{p_data['name'].lower().replace(' ', '_')}",
                    embedding_provider=self.embedding_provider,
                    embedding_model=self.embedding_model
                )
                
                # Create a knowledge base
                knowledge_base = KnowledgeBase(
                    vector_db=vector_db,
                    name=f"{p_data['name']} Knowledge Base"
                )
                
                # Create the personality
                personality = Personality.from_dict(
                    data=p_data,
                    knowledge_base=knowledge_base
                )
                
                # Add to personalities
                self.personalities[p_id] = personality
            
            # Set active personality
            self.active_personality_id = data.get("active_personality_id")
            
            logger.info(f"Loaded {len(self.personalities)} personalities from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading personalities: {e}")
            return False 