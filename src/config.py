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
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")

    # ------------------------------------------------------------------
    # LLM Configuration - multi-provider cascade
    #
    # Providers are tried in PROVIDER_ORDER. Within each provider, its models
    # are tried in list order. The first model that returns a response wins.
    # A provider is skipped entirely if its API key is missing.
    # ------------------------------------------------------------------
    PROVIDER_ORDER: list = ["groq", "google", "openrouter"]

    # Groq models (native Groq API) - requires GROQ_API_KEY
    GROQ_MODELS: list = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
        "llama-3.1-8b-instant",
    ]

    # Google Gemini models (native Google AI API) - requires GOOGLE_API_KEY
    GOOGLE_MODELS: list = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]

    # OpenRouter models - requires OPENROUTER_API_KEY. OPENROUTER_MODEL is tried
    # first, then OPENROUTER_FALLBACK_MODELS in order.
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_FALLBACK_MODELS: list = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nex-agi/nex-n2-pro:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ]
    OPENROUTER_MAX_RETRIES: int = 0  # No client-level retries - cascade handles it

    EMBEDDING_MODEL: str = "nvidia/nv-embedqa-e5-v5"
    EMBEDDING_TRUNCATE: str = "END"
    
    # Pinecone Settings
    PINECONE_INDEX_NAME: str = "leee-helpbot-nvidia-embeddings"
    # Namespace this bot owns. All upsert/update/delete operations are scoped to
    # this namespace ONLY. The default Pinecone namespace ("") is where this bot's
    # data lives; other namespaces (e.g. "leee-vercel", used by another bot) are
    # never modified.
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "")
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

        # Provider keys are optional; a provider in PROVIDER_ORDER is skipped if
        # its key is missing. Warn so misconfiguration is visible.
        provider_keys = {'groq': cls.GROQ_API_KEY, 'google': cls.GOOGLE_API_KEY}
        for provider in cls.PROVIDER_ORDER:
            if provider in provider_keys and not provider_keys[provider]:
                logger.warning(
                    f"{provider.upper()}_API_KEY not set. '{provider}' provider will be skipped in the cascade."
                )
        
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
            'provider_order': cls.PROVIDER_ORDER,
            'groq_models': cls.GROQ_MODELS,
            'google_models': cls.GOOGLE_MODELS,
            'openrouter_model': cls.OPENROUTER_MODEL,
            'openrouter_fallbacks': cls.OPENROUTER_FALLBACK_MODELS,
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
