"""
Query service with caching, validation, error handling, and retry logic
"""
import asyncio
import time
from typing import Optional
import logging

from llama_index.core.settings import Settings
from llama_index.core.prompts import PromptTemplate
from openai import RateLimitError

from src.config import Config
from src.services.embedding import get_embedding_service
from src.utils.cache import get_cache_manager
from src.utils.validators import create_query_validator
from src.utils.metrics import metrics, MetricsContext

logger = logging.getLogger(__name__)


# Define custom QA template
QA_TEMPLATE = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using ONLY the context information above, answer the question concisely. "
    "List only the specific topics mentioned in the context. "
    "Do NOT add any information not present in the context. "
    "If the context doesn't answer the question, say 'No relevant information found in the documents.'\n"
    "Question: {query_str}\n"
    "Answer: "
)


class QueryService:
    """
    Service for handling user queries with caching and validation
    """
    
    def __init__(self):
        """Initialize query service"""
        self.embedding_service = get_embedding_service()
        self.cache_manager = get_cache_manager(
            response_cache_size=Config.MAX_CACHE_SIZE,
            cache_ttl=Config.CACHE_TTL_SECONDS
        )
        self.validator = create_query_validator(
            max_length=Config.MAX_QUERY_LENGTH,
            min_length=Config.MIN_QUERY_LENGTH
        )
        self.initialized = False
        self._init_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize query service dependencies"""
        if self.initialized:
            return
        
        async with self._init_lock:
            if self.initialized:
                return
            
            try:
                await self.embedding_service.initialize()
                self.initialized = True
                logger.info("QueryService initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize QueryService: {e}", exc_info=True)
                raise
    
    async def process_query(self, input_text: str) -> str:
        """
        Process user query with validation, caching, and error handling
        
        Args:
            input_text: User's query text
            
        Returns:
            Response text
            
        Raises:
            ValueError: If query validation fails
            Exception: If query processing fails
        """
        start_time = time.time()
        from_cache = False
        
        try:
            # Ensure service is initialized
            if not self.initialized:
                await self.initialize()
            
            # Sanitize and validate input
            input_text = self.validator.sanitize(input_text)
            is_valid, error_msg = self.validator.validate(input_text)
            
            if not is_valid:
                logger.warning(f"Query validation failed: {error_msg}")
                raise ValueError(error_msg)
            
            # Check appropriateness
            is_appropriate, reason = self.validator.is_appropriate(input_text)
            if not is_appropriate:
                logger.warning(f"Inappropriate query rejected: {reason}")
                return self._get_inappropriate_response()
            
            # Check cache
            if Config.ENABLE_RESPONSE_CACHE:
                cached_response = await self.cache_manager.get_cached_response(input_text)
                if cached_response:
                    from_cache = True
                    response = cached_response
                    logger.info(f"Returning cached response for query: {input_text[:50]}...")
                else:
                    response = await self._execute_query(input_text)
                    # Cache the response
                    await self.cache_manager.set_cached_response(input_text, response)
            else:
                response = await self._execute_query(input_text)
            
            # Record metrics
            elapsed_time = time.time() - start_time
            metrics.record_query(success=True, response_time=elapsed_time, from_cache=from_cache)
            
            logger.info(f"Query processed successfully in {elapsed_time:.3f}s (cached: {from_cache})")
            return response
            
        except ValueError as e:
            # Validation error - don't retry
            elapsed_time = time.time() - start_time
            metrics.record_query(success=False, response_time=elapsed_time)
            logger.warning(f"Query validation error: {e}")
            raise
            
        except Exception as e:
            # Other errors - record and re-raise
            elapsed_time = time.time() - start_time
            metrics.record_query(success=False, response_time=elapsed_time)
            metrics.record_error(type(e).__name__)
            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise
    
    async def _execute_query(self, input_text: str) -> str:
        """
        Execute query with retry logic
        
        Args:
            input_text: User's query text
            
        Returns:
            Response text
        """
        # Try to get cached index first
        index = await self.cache_manager.get_cached_index()
        
        if index is None:
            # Load index and cache it
            logger.info("Index not cached, loading from Pinecone...")
            index = await self.embedding_service.load_index(Config.DATA_DIR)
            await self.cache_manager.set_cached_index(index)
        else:
            logger.debug("Using cached index")
        
        # Retry logic for query execution
        last_exception = None
        for attempt in range(Config.MAX_RETRIES):
            try:
                return await self._run_query(index, input_text)
            except Exception as e:
                last_exception = e
                error_msg = f"Query execution attempt {attempt + 1}/{Config.MAX_RETRIES} failed: {e}"
                logger.error(error_msg)
                
                if attempt < Config.MAX_RETRIES - 1:
                    delay = Config.RETRY_DELAY_SECONDS * (Config.RETRY_BACKOFF_FACTOR ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
        
        # All retries failed
        raise last_exception
    
    async def _run_query(self, index, input_text: str) -> str:
        """
        Run query against index with Groq fallback
        
        Args:
            index: VectorStoreIndex instance
            input_text: User's query text
            
        Returns:
            Response text
        """
        with MetricsContext("retrieval"):
            # Get retriever and retrieve relevant chunks
            retriever = index.as_retriever(similarity_top_k=Config.SIMILARITY_TOP_K)
            nodes = await retriever.aretrieve(input_text)
        
        # Extract all retrieved text with source info
        retrieved_chunks = []
        source_files = set()
        for i, node in enumerate(nodes, 1):
            source_name = node.metadata.get('file_name', 'Unknown')
            source_files.add(source_name)
            retrieved_chunks.append(f"[Source {i}: {source_name}]\n{node.text}")
        
        retrieved_text = "\n\n---\n\n".join(retrieved_chunks)
        
        # Check if we got relevant results
        if not retrieved_text.strip():
            logger.warning(f"No relevant documents found for query: {input_text[:50]}...")
            return self._get_no_results_response()
        
        # Create prompt with all retrieved content
        prompt = self._build_prompt(input_text, retrieved_text)
        
        # Query LLM with Groq fallback on rate limit
        with MetricsContext("llm_query"):
            try:
                # Try primary LLM (OpenRouter)
                response = await Settings.llm.acomplete(prompt)
                response_text = response.text
                logger.debug(f"LLM response from OpenRouter, length: {len(response_text)} characters")
            except RateLimitError as e:
                logger.warning(f"OpenRouter rate limit hit: {e}")
                
                # Try Groq fallback if available
                primary_llm, fallback_llm = self.embedding_service.get_llms()
                
                if fallback_llm and Config.ENABLE_GROQ_FALLBACK:
                    logger.info("Attempting fallback to Groq...")
                    try:
                        # Temporarily switch to Groq
                        original_llm = Settings.llm
                        Settings.llm = fallback_llm
                        
                        response = await Settings.llm.acomplete(prompt)
                        response_text = response.text
                        
                        # Restore original LLM
                        Settings.llm = original_llm
                        
                        logger.info(f"Successfully used Groq fallback, response length: {len(response_text)} characters")
                    except Exception as fallback_error:
                        logger.error(f"Groq fallback also failed: {fallback_error}")
                        # Restore original LLM
                        Settings.llm = original_llm
                        raise Exception(
                            "Both OpenRouter and Groq rate limits exceeded. Please try again later."
                        )
                else:
                    logger.error("No fallback LLM available or fallback disabled")
                    raise Exception(
                        "OpenRouter rate limit exceeded and no fallback available. Please try again later."
                    )
        
        logger.debug(f"Final response length: {len(response_text)} characters")
        return response_text
    
    def _build_prompt(self, query: str, retrieved_text: str) -> str:
        """
        Build prompt for LLM
        
        Args:
            query: User's query
            retrieved_text: Retrieved context
            
        Returns:
            Complete prompt
        """
        return f"""You are a helpful assistant answering questions about IIIT Hyderabad's Lateral Entry Exam (LEEE) and related academic programs.

Retrieved Information from Knowledge Base:
{retrieved_text}

User Question: {query}

Instructions:
- First, check if the question is relevant to LEEE, IIIT Hyderabad, or related academic topics
- If the question contains inappropriate language, curse words, or is completely unrelated to LEEE/IIITH, respond politely: "I'm designed to answer questions about IIIT Hyderabad's LEEE program. Please check the #resources channel for comprehensive information."
- If relevant, use the information from the retrieved content above to provide a comprehensive answer
- Answer the question directly and thoroughly
- Include ALL relevant information that answers the user's question:
  * ALL URLs and links exactly as they appear in the retrieved content (if any)
  * YouTube channel names and playlists with their URLs (if any)
  * Book recommendations with purchase links (if any)
  * Course details, syllabus information from PDFs (if any)
  * Online resource links (NPTEL, GeeksforGeeks, etc.) (if any)
  * Interview experiences and tips from Quora/Medium articles (if any)
- Preserve the original formatting of links (markdown format [text](url) or plain URLs)
- If multiple sources provide similar information, synthesize them into a comprehensive answer
- If the retrieved content doesn't contain the answer, say: "I don't have specific information about this in my knowledge base. Please check the #resources channel for comprehensive LEEE information."

Provide a well-structured, informative response:

Answer:"""
    
    def _get_inappropriate_response(self) -> str:
        """Get response for inappropriate queries"""
        return (
            "I'm designed to answer questions about IIIT Hyderabad's LEEE program. "
            "Please check the #resources channel for comprehensive information."
        )
    
    def _get_no_results_response(self) -> str:
        """Get response when no relevant documents found"""
        return (
            "I don't have specific information about this. "
            "Please check the #resources channel for comprehensive LEEE information."
        )
    
    async def get_stats(self) -> dict:
        """
        Get query service statistics
        
        Returns:
            Dictionary with service statistics
        """
        cache_stats = await self.cache_manager.get_cache_stats()
        index_stats = await self.embedding_service.get_index_stats()
        
        return {
            'initialized': self.initialized,
            'cache': cache_stats,
            'index': index_stats,
            'config': {
                'similarity_top_k': Config.SIMILARITY_TOP_K,
                'max_query_length': Config.MAX_QUERY_LENGTH,
                'cache_enabled': Config.ENABLE_RESPONSE_CACHE,
            }
        }


# Global service instance
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """
    Get or create global query service instance
    
    Returns:
        QueryService instance
    """
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service


# Backward compatibility function
async def data_querying(input_text: str) -> str:
    """
    Process a query (backward compatibility wrapper)
    
    Args:
        input_text: User's query
        
    Returns:
        Response text
    """
    service = get_query_service()
    return await service.process_query(input_text)
