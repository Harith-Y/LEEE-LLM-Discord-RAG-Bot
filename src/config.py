"""
Configuration management for LEEE Discord RAG Bot
Centralizes all configuration and validates environment variables
"""
import os
from typing import Optional
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration class"""
    
    # Discord Configuration
    DISCORD_BOT_TOKEN: Optional[str] = os.getenv("DISCORD_BOT_TOKEN")
    
    # API Keys
    NVIDIA_API_KEY: Optional[str] = os.getenv("NVIDIA_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    
    # LLM Configuration
    GROQ_MODEL: str = "llama-3.3-70b-versatile"  # Groq primary
    LLM_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"  # OpenRouter fallback
    EMBEDDING_MODEL: str = "nvidia/nv-embedqa-e5-v5"
    EMBEDDING_TRUNCATE: str = "END"
    ENABLE_OPENROUTER_FALLBACK: bool = True  # Enable OpenRouter fallback on rate limits
    
    # Pinecone Settings
    PINECONE_INDEX_NAME: str = "leee-helpbot-nvidia-embeddings"
    PINECONE_DIMENSION: int = 1024  # NVIDIA nv-embedqa-e5-v5 dimension
    PINECONE_METRIC: str = "cosine"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    
    # Retrieval Settings
    SIMILARITY_TOP_K: int = 15  # Increased for better coverage of expanded dataset
    MAX_RESPONSE_LENGTH: int = 2000  # Discord standard message limit
    
    # Rate Limiting
    RATE_LIMIT_MAX_REQUESTS: int = 5
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    # Caching
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    ENABLE_RESPONSE_CACHE: bool = True
    MAX_CACHE_SIZE: int = 1000  # Maximum number of cached responses
    
    # Validation
    MAX_QUERY_LENGTH: int = 500
    MIN_QUERY_LENGTH: int = 3
    
    # Paths
    DATA_DIR: str = "data"
    LOG_DIR: str = "logs"
    
    # Server
    PORT: int = int(os.getenv('PORT', 8080))
    HOST: str = '0.0.0.0'
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    RETRY_BACKOFF_FACTOR: float = 2.0
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate required environment variables are set
        
        Raises:
            ValueError: If required environment variables are missing
        """
        required_vars = {
            'DISCORD_BOT_TOKEN': cls.DISCORD_BOT_TOKEN,
            'NVIDIA_API_KEY': cls.NVIDIA_API_KEY,
            'OPENROUTER_API_KEY': cls.OPENROUTER_API_KEY,
            'PINECONE_API_KEY': cls.PINECONE_API_KEY,
        }
        
        # OPENROUTER_API_KEY is optional (only needed if fallback is enabled)
        if cls.ENABLE_OPENROUTER_FALLBACK and not cls.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set. OpenRouter fallback will be disabled.")
        
        missing = [key for key, value in required_vars.items() if not value]
        
        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation successful")
    
    @classmethod
    def get_summary(cls) -> dict:
        """
        Get a summary of non-sensitive configuration
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'llm_model': cls.LLM_MODEL,
            'embedding_model': cls.EMBEDDING_MODEL,
            'pinecone_index': cls.PINECONE_INDEX_NAME,
            'similarity_top_k': cls.SIMILARITY_TOP_K,
            'rate_limit': f"{cls.RATE_LIMIT_MAX_REQUESTS} requests per {cls.RATE_LIMIT_WINDOW_SECONDS}s",
            'cache_enabled': cls.ENABLE_RESPONSE_CACHE,
            'cache_ttl': f"{cls.CACHE_TTL_SECONDS}s",
            'max_query_length': cls.MAX_QUERY_LENGTH,
        }


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    logger.warning(f"Configuration validation failed: {e}")
    logger.warning("Bot may not function correctly without required environment variables")
