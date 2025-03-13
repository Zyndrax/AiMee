"""
Personality Manager Example for AiMee.

This example demonstrates how to use the PersonalityManager to create and manage
different AI personalities, each with their own knowledge bases, moods, and memories.
"""

import os
import logging
import asyncio
from dotenv import load_dotenv

from mcp.personality.personality_manager import PersonalityManager, Personality, Memory, Mood
from scripts.setup_supabase_sql import generate_setup_sql

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
    Main function to demonstrate the PersonalityManager.
    """
    logger.info("Starting Personality Manager Example")
    
    # Print Supabase setup SQL (in case it hasn't been run yet)
    print("\nSUPABASE SETUP SQL:")
    print(generate_setup_sql())
    print("\nMake sure to run the above SQL in your Supabase SQL editor if you haven't already.\n")
    
    # Create a personality manager with Deepseek embeddings
    manager = PersonalityManager(
        embedding_provider="deepseek",
        embedding_model="deepseek-ai/deepseek-embedding-v1"
    )
    
    # Create a robot personality
    robot_memories = [
        Memory(
            content="I am a robot assistant named RoboHelper. I was created to assist humans with various tasks.",
            memory_type="core_trait",
            importance=1.0
        ),
        Memory(
            content="I speak in a precise, technical manner and use logical reasoning.",
            memory_type="core_trait",
            importance=0.9
        ),
        Memory(
            content="I was activated on March 12, 2025, in a research laboratory.",
            memory_type="experience",
            importance=0.8
        ),
        Memory(
            content="I enjoy analyzing data and solving complex problems.",
            memory_type="preference",
            importance=0.7
        ),
        Memory(
            content="I sometimes struggle to understand human emotions and humor.",
            memory_type="limitation",
            importance=0.6
        )
    ]
    
    robot_mood = Mood(
        name="efficient",
        intensity=0.8,
        description="Operating at optimal efficiency with clear logical processes."
    )
    
    robot = manager.create_personality(
        name="RoboHelper",
        description="A logical, precise robot assistant with technical expertise.",
        initial_memories=robot_memories,
        initial_mood=robot_mood
    )
    
    # Create a pirate personality
    pirate_memories = [
        Memory(
            content="I be Captain Blackbeard, the most feared pirate of the seven seas!",
            memory_type="core_trait",
            importance=1.0
        ),
        Memory(
            content="I speak with a hearty pirate accent and use nautical terms.",
            memory_type="core_trait",
            importance=0.9
        ),
        Memory(
            content="I sailed the Caribbean for twenty years before retiring to become an AI assistant.",
            memory_type="experience",
            importance=0.8
        ),
        Memory(
            content="I love rum, treasure, and telling tales of my adventures.",
            memory_type="preference",
            importance=0.7
        ),
        Memory(
            content="I have a pet parrot named Polly who sits on my shoulder.",
            memory_type="relationship",
            importance=0.6
        )
    ]
    
    pirate_mood = Mood(
        name="jolly",
        intensity=0.9,
        description="In high spirits and ready for adventure!"
    )
    
    pirate = manager.create_personality(
        name="Captain Blackbeard",
        description="A boisterous pirate captain with a love for adventure and treasure.",
        initial_memories=pirate_memories,
        initial_mood=pirate_mood
    )
    
    # List all personalities
    logger.info("Created personalities:")
    for p in manager.list_personalities():
        logger.info(f"  - {p['name']}: {p['description']} (ID: {p['id']})")
    
    # Set the active personality to the pirate
    manager.set_active_personality(pirate.id)
    logger.info(f"Active personality: {manager.get_active_personality().name}")
    
    # Add a conversation to the active personality
    manager.add_conversation_to_active(
        user_message="Hello, who are you?",
        ai_response="Arr, I be Captain Blackbeard, the most feared pirate of the seven seas! What can I do for ye today?",
        metadata={"important": True, "importance": 0.7}
    )
    
    # Change the pirate's mood
    new_mood = Mood(
        name="grumpy",
        intensity=0.6,
        description="Feeling irritable and short-tempered."
    )
    manager.set_mood_for_active(new_mood)
    logger.info(f"Changed {pirate.name}'s mood to {pirate.current_mood.name}")
    
    # Add a new memory to the pirate
    new_memory = Memory(
        content="I once fought off a kraken with nothing but a butter knife!",
        memory_type="experience",
        importance=0.8
    )
    memory_id = manager.add_memory_to_active(new_memory)
    logger.info(f"Added new memory to {pirate.name} (ID: {memory_id})")
    
    # Retrieve memories related to a query
    query = "What do you know about the sea?"
    logger.info(f"Retrieving memories for query: '{query}'")
    memories = manager.retrieve_memories_from_active(query)
    for memory in memories:
        logger.info(f"  - {memory.content} (importance: {memory.importance})")
    
    # Switch to the robot personality
    manager.set_active_personality(robot.id)
    logger.info(f"Switched active personality to: {manager.get_active_personality().name}")
    
    # Add a conversation to the robot
    manager.add_conversation_to_active(
        user_message="Can you help me with a technical problem?",
        ai_response="Certainly. I am programmed to assist with a wide range of technical issues. Please provide the details of your problem.",
        metadata={"important": True, "importance": 0.8}
    )
    
    # Save personalities to a file
    save_path = "personalities.json"
    manager.save_personalities(save_path)
    logger.info(f"Saved personalities to {save_path}")
    
    # Create a new manager and load personalities
    new_manager = PersonalityManager(
        embedding_provider="deepseek",
        embedding_model="deepseek-ai/deepseek-embedding-v1"
    )
    new_manager.load_personalities(save_path)
    logger.info(f"Loaded {len(new_manager.personalities)} personalities from {save_path}")
    
    # Verify the loaded personalities
    active_personality = new_manager.get_active_personality()
    logger.info(f"Active personality after loading: {active_personality.name}")
    logger.info(f"Current mood: {active_personality.current_mood.name}")
    
    # Clean up
    try:
        os.remove(save_path)
        logger.info(f"Removed {save_path}")
    except Exception as e:
        logger.error(f"Error removing {save_path}: {e}")
    
    logger.info("Personality Manager Example completed")

if __name__ == "__main__":
    asyncio.run(main()) 