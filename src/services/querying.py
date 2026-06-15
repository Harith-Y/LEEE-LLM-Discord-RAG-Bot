"""
Query service with caching, validation, error handling, and retry logic
"""
import asyncio
import time
from typing import Optional
import logging

from llama_index.core.settings import Settings
from llama_index.core.prompts import PromptTemplate

from src.config import Config
from src.services.embedding import get_embedding_service
from src.utils.cache import get_cache_manager
from src.utils.validators import create_query_validator
from src.utils.metrics import metrics, MetricsContext
from src.utils.discord_formatter import format_for_discord

logger = logging.getLogger(__name__)


# Define custom QA template - instructions come BEFORE context so the model
# reads them before generating its answer (not after "Answer:" where they're ignored).
QA_TEMPLATE = PromptTemplate(
    "You are a helpful and knowledgeable assistant answering questions about IIIT Hyderabad's Lateral Entry Exam (LEEE) and related academic programs.\n\n"
    "**Response Instructions:**\n"
    "1. ONLY answer if the question is relevant to LEEE, IIIT Hyderabad, or related academic topics. "
    "If it is completely unrelated, inappropriate, or uses curse words, output strictly: "
    "\"I'm designed to answer questions about IIIT Hyderabad's LEEE program. Please check the #resources channel for comprehensive information.\"\n"
    "2. Base your answer primarily on the Context Information provided below.\n"
    "3. Answer the question directly, naturally, and comprehensively in an empathetic tone. "
    "Synthesize the provided information into a directly helpful response. "
    "Be thorough — include ALL relevant details, tips, syllabus topics, dates, and resources from the context. "
    "Your response should be detailed and complete, covering every relevant point found in the context. Do NOT give a brief or summarized answer.\n"
    "4. NEVER mention source names, file names, or source numbers "
    "(e.g. do NOT say \"According to [Source 1]\" or cite any filename). Just synthesize the information naturally.\n"
    "5. Retain ALL relevant information: resources, tips, syllabus topics, and exact links/URLs (do not summarize links away).\n"
    "6. Preserve the original formatting of links (markdown format [text](url) or plain URLs).\n"
    "7. Do NOT use Markdown tables. Use numbered lists, bullet points, or bold headers instead.\n"
    "8. Do NOT use HTML tags (like <br>, <b>, <i>, etc.). Use pure Markdown.\n"
    "9. If the Context Information does not contain the answer, say: "
    "\"I don't have specific information about this in my knowledge base. Please check the #resources channel for comprehensive LEEE information.\"\n\n"
    "**Context Information:**\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "**Question:** {query_str}\n\n"
    "**Answer:**"
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
        
        # Query the multi-provider LLM cascade. Models are tried in the order
        # built from Config.PROVIDER_ORDER; the first to respond wins.
        LLM_TIMEOUT = 90  # seconds — free-tier models are slow; give enough time for long responses

        with MetricsContext("llm_query"):
            llm_chain = self.embedding_service.get_llm_chain()
            original_llm = Settings.llm
            response_text = None
            last_error = None

            for provider, model_name, llm in llm_chain:
                try:
                    logger.info(f"Attempting {provider} model: {model_name}...")
                    Settings.llm = llm
                    response = await asyncio.wait_for(
                        llm.acomplete(prompt), timeout=LLM_TIMEOUT
                    )
                    response_text = response.text
                    logger.info(
                        f"Successfully used {provider}:{model_name}, "
                        f"response length: {len(response_text)} characters"
                    )
                    break  # Success, exit cascade
                except Exception as e:
                    last_error = e
                    logger.warning(f"{provider} model '{model_name}' failed: {e}")
                    continue  # Try next model in the cascade
                finally:
                    Settings.llm = original_llm

            if response_text is None:
                logger.error(f"All {len(llm_chain)} models in the LLM cascade failed. Last error: {last_error}")
                raise Exception(
                    "All LLM providers failed or rate limited. Please try again later."
                )

        # Post-process for Discord compatibility (tables, <br>, etc.)
        response_text = format_for_discord(response_text)
        logger.debug(f"Final response length: {len(response_text)} characters")
        return response_text
    
    def _build_prompt(self, query: str, retrieved_text: str) -> str:
        return QA_TEMPLATE.format(context_str=retrieved_text, query_str=query)
        
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
