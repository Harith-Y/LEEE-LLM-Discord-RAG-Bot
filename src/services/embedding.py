"""
Embedding service with error handling, retry logic, and incremental updates
"""
import asyncio
import time
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging

from llama_index.core.settings import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core import StorageContext, SimpleDirectoryReader, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.groq import Groq
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from pinecone import Pinecone, ServerlessSpec

from src.config import Config
from src.utils.metrics import metrics, MetricsContext

logger = logging.getLogger(__name__)

try:
    from llama_index.readers.file import PDFReader
    PDF_READER_AVAILABLE = True
except ImportError:
    PDF_READER_AVAILABLE = False
    logger.warning("llama-index-readers-file not available. Using default PDF reader.")


class EmbeddingService:
    """
    Service for managing document embeddings with error handling and retry logic
    """
    
    def __init__(self):
        """Initialize embedding service"""
        self.pc: Optional[Pinecone] = None
        self.pinecone_index = None
        self.initialized = False
        self._init_lock = asyncio.Lock()
        self.document_hashes_file = Path(Config.DATA_DIR) / ".document_hashes.json"
        self.primary_llm = None  # Groq
        self.fallback_llm = None  # OpenRouter
    
    async def initialize(self) -> None:
        """
        Initialize Pinecone and LLM with retry logic
        
        Raises:
            Exception: If initialization fails after retries
        """
        if self.initialized:
            return
        
        async with self._init_lock:
            if self.initialized:
                return
            
            for attempt in range(Config.MAX_RETRIES):
                try:
                    await self._initialize_llm()
                    await self._initialize_pinecone()
                    self.initialized = True
                    logger.info("EmbeddingService initialized successfully")
                    return
                except Exception as e:
                    error_msg = f"Initialization attempt {attempt + 1}/{Config.MAX_RETRIES} failed: {e}"
                    logger.error(error_msg)
                    
                    if attempt < Config.MAX_RETRIES - 1:
                        delay = Config.RETRY_DELAY_SECONDS * (Config.RETRY_BACKOFF_FACTOR ** attempt)
                        logger.info(f"Retrying in {delay:.1f} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        logger.critical("Failed to initialize EmbeddingService after all retries")
                        metrics.record_error("InitializationError")
                        raise
    
    async def _initialize_llm(self) -> None:
        """Initialize LLM and embedding model"""
        logger.info(f"Initializing NVIDIA embeddings with {Config.EMBEDDING_MODEL}...")
        embed_model = NVIDIAEmbedding(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.NVIDIA_API_KEY,
            truncate=Config.EMBEDDING_TRUNCATE
        )
        
        logger.info(f"Initializing OpenRouter (primary) with {Config.OPENROUTER_MODEL}...")
        self.primary_llm = OpenRouter(
            model=Config.OPENROUTER_MODEL,
            api_key=Config.OPENROUTER_API_KEY,
            max_tokens=2048  # Increased token limit for complete responses
        )
        
        # Initialize Groq as fallback if enabled and API key is available
        if Config.ENABLE_GROQ_FALLBACK and Config.GROQ_API_KEY:
            logger.info(f"Initializing Groq fallback with {Config.GROQ_MODEL}...")
            try:
                self.fallback_llm = Groq(
                    model=Config.GROQ_MODEL,
                    api_key=Config.GROQ_API_KEY,
                    max_tokens=2048
                )
                logger.info("Groq fallback initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq fallback: {e}")
                self.fallback_llm = None
        else:
            logger.info("OpenRouter fallback disabled or API key not provided")
            self.fallback_llm = None
        
        Settings.llm = self.primary_llm
        Settings.embed_model = embed_model
        logger.info("LLM and embeddings initialized successfully")
    
    async def _initialize_pinecone(self) -> None:
        """Initialize Pinecone connection and index"""
        logger.info("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Create index if it doesn't exist
        if Config.PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            logger.info(f"Creating Pinecone index '{Config.PINECONE_INDEX_NAME}'...")
            self.pc.create_index(
                name=Config.PINECONE_INDEX_NAME,
                dimension=Config.PINECONE_DIMENSION,
                metric=Config.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=Config.PINECONE_CLOUD,
                    region=Config.PINECONE_REGION
                )
            )
            
            # Wait for index to be ready
            max_wait = 60  # seconds
            start_time = time.time()
            while not self.pc.describe_index(Config.PINECONE_INDEX_NAME).status['ready']:
                if time.time() - start_time > max_wait:
                    raise TimeoutError("Pinecone index creation timed out")
                await asyncio.sleep(1)
            
            logger.info(f"Pinecone index '{Config.PINECONE_INDEX_NAME}' created successfully")
        else:
            logger.info(f"Pinecone index '{Config.PINECONE_INDEX_NAME}' already exists")
        
        self.pinecone_index = self.pc.Index(Config.PINECONE_INDEX_NAME)
    
    def get_llms(self) -> tuple:
        """
        Get primary and fallback LLMs
        
        Returns:
            Tuple of (primary_llm, fallback_llm)
        """
        return (self.primary_llm, self.fallback_llm)
    
    async def load_index(self, directory_path: str = None) -> VectorStoreIndex:
        """
        Load or create index from Pinecone
        
        Args:
            directory_path: Path to documents directory (default: Config.DATA_DIR)
            
        Returns:
            VectorStoreIndex instance
            
        Raises:
            Exception: If index loading fails
        """
        if not self.initialized:
            await self.initialize()
        
        if directory_path is None:
            directory_path = Config.DATA_DIR
        
        with MetricsContext("load_index"):
            try:
                documents = SimpleDirectoryReader(directory_path).load_data()
                logger.info(f"Loaded {len(documents)} documents from {directory_path}")
                
                # Check if index has data
                stats = self.pinecone_index.describe_index_stats()
                
                if stats['total_vector_count'] > 0:
                    logger.info(f"Loading existing index from Pinecone ({stats['total_vector_count']} vectors)...")
                    vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
                    index = VectorStoreIndex.from_vector_store(vector_store)
                    logger.info("Index loaded from Pinecone (no re-embedding needed)")
                else:
                    logger.info("No existing embeddings found. Creating new index...")
                    index = await self._create_new_index(documents)
                    logger.info("New index created and stored in Pinecone")
                
                return index
                
            except Exception as e:
                logger.error(f"Failed to load index: {e}", exc_info=True)
                metrics.record_error(type(e).__name__)
                raise
    
    async def _create_new_index(self, documents: List[Document]) -> VectorStoreIndex:
        """
        Create new index from documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            VectorStoreIndex instance
        """
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(
            None,
            lambda: VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
        )
        
        return index
    
    async def update_index(self, directory_path: str = None, incremental: bool = True) -> Optional[int]:
        """
        Update index with new/changed documents
        
        Args:
            directory_path: Path to documents directory (default: Config.DATA_DIR)
            incremental: If True, only update changed documents; if False, full rebuild
            
        Returns:
            Number of documents processed, or None on error
        """
        if not self.initialized:
            await self.initialize()
        
        if directory_path is None:
            directory_path = Config.DATA_DIR
        
        with MetricsContext("update_index"):
            try:
                if incremental:
                    return await self._incremental_update(directory_path)
                else:
                    return await self._full_update(directory_path)
                    
            except Exception as e:
                logger.error(f"Failed to update index: {e}", exc_info=True)
                metrics.record_error(type(e).__name__)
                return None
    
    async def _full_update(self, directory_path: str) -> int:
        """
        Perform full index update (clear and rebuild)
        
        Args:
            directory_path: Path to documents directory
            
        Returns:
            Number of documents processed
        """
        logger.info("Starting full index update...")
        logger.info(f"Scanning directory: {directory_path}")
        
        # Load documents with progress logging
        from pathlib import Path
        import re
        data_path = Path(directory_path)
        all_files = list(data_path.glob('*'))
        logger.info(f"Found {len(all_files)} files to process")
        
        # Configure file extractors for better PDF parsing
        file_extractor = None
        if PDF_READER_AVAILABLE:
            from llama_index.readers.file import PDFReader
            file_extractor = {".pdf": PDFReader()}
            logger.info("Using PyMuPDF reader for enhanced PDF processing")
        
        documents = SimpleDirectoryReader(
            directory_path,
            filename_as_id=True,
            required_exts=[".txt", ".md", ".pdf"],
            file_extractor=file_extractor
        ).load_data()
        
        # Sanitize document IDs for Pinecone (ASCII-only, no special chars)
        for doc in documents:
            if doc.id_:
                # Extract just the filename from the path
                filename = Path(doc.id_).name
                # Remove non-ASCII characters and special chars
                sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
                # Add unique suffix if needed
                doc.id_ = f"{sanitized}_{doc.hash}"
        
        logger.info(f"Loaded {len(documents)} documents for update")
        
        # Clear all existing vectors
        logger.info("Clearing existing vectors from Pinecone...")
        stats = self.pinecone_index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        
        if not namespaces:
            namespaces = ['']  # Default namespace
        
        for ns in namespaces:
            self.pinecone_index.delete(delete_all=True, namespace=ns)
        
        logger.info("Existing vectors cleared. Creating fresh embeddings...")
        
        # Create fresh embeddings
        await self._create_new_index(documents)
        
        # Update document hashes
        await self._save_document_hashes(documents)
        
        # Verify upload
        stats = self.pinecone_index.describe_index_stats()
        logger.info(f"Index updated successfully. Total vectors: {stats['total_vector_count']}")
        
        return len(documents)
    
    async def _incremental_update(self, directory_path: str) -> int:
        """
        Perform incremental update (only changed/new documents)
        
        Args:
            directory_path: Path to documents directory
            
        Returns:
            Number of documents updated
        """
        logger.info("Starting incremental index update...")
        from pathlib import Path
        import re
        
        # Configure file extractors for better PDF parsing
        file_extractor = None
        if PDF_READER_AVAILABLE:
            from llama_index.readers.file import PDFReader
            file_extractor = {".pdf": PDFReader()}
            logger.info("Using PyMuPDF reader for enhanced PDF processing")
        
        # Load current documents
        documents = SimpleDirectoryReader(
            directory_path,
            filename_as_id=True,
            required_exts=[".txt", ".md", ".pdf"],
            file_extractor=file_extractor
        ).load_data()
        
        # Sanitize document IDs for Pinecone (ASCII-only, no special chars)
        for doc in documents:
            if doc.id_:
                # Extract just the filename from the path
                filename = Path(doc.id_).name
                # Remove non-ASCII characters and special chars
                sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
                # Add unique suffix if needed
                doc.id_ = f"{sanitized}_{doc.hash}"
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Load previous document hashes
        old_hashes = await self._load_document_hashes()
        
        # Compute current hashes and find changes
        new_hashes = {}
        changed_docs = []
        
        for doc in documents:
            doc_hash = self._compute_document_hash(doc)
            doc_id = self._get_document_id(doc)
            new_hashes[doc_id] = doc_hash
            
            if doc_id not in old_hashes or old_hashes[doc_id] != doc_hash:
                changed_docs.append(doc)
                logger.debug(f"Document changed: {doc_id}")
        
        # Find deleted documents
        deleted_ids = set(old_hashes.keys()) - set(new_hashes.keys())
        
        if not changed_docs and not deleted_ids:
            logger.info("No changes detected. Index is up to date.")
            return 0
        
        logger.info(f"Found {len(changed_docs)} changed/new documents and {len(deleted_ids)} deleted documents")
        
        # Delete removed documents from index
        if deleted_ids:
            logger.info(f"Deleting {len(deleted_ids)} removed documents...")
            # Note: This is a simplified version. In production, you'd need to track vector IDs
            for doc_id in deleted_ids:
                logger.debug(f"Document deleted: {doc_id}")
        
        # Update changed documents
        if changed_docs:
            logger.info(f"Updating {len(changed_docs)} changed/new documents...")
            vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
            index = VectorStoreIndex.from_vector_store(vector_store)
            
            # Add new documents to index
            for doc in changed_docs:
                index.insert(doc)
        
        # Save new document hashes
        await self._save_document_hashes(documents)
        
        logger.info(f"Incremental update complete. Updated {len(changed_docs)} documents.")
        return len(changed_docs)
    
    def _compute_document_hash(self, document: Document) -> str:
        """
        Compute hash of document content
        
        Args:
            document: Document to hash
            
        Returns:
            MD5 hash of document content
        """
        content = document.get_content() + str(document.metadata)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_document_id(self, document: Document) -> str:
        """
        Get unique identifier for document
        
        Args:
            document: Document
            
        Returns:
            Document ID (from metadata or hash)
        """
        # Try to get filename from metadata
        if 'file_name' in document.metadata:
            return document.metadata['file_name']
        elif 'source' in document.metadata:
            return document.metadata['source']
        else:
            # Fallback to hash if no metadata available
            return self._compute_document_hash(document)
    
    async def _save_document_hashes(self, documents: List[Document]) -> None:
        """
        Save document hashes to file
        
        Args:
            documents: List of documents
        """
        hashes = {}
        for doc in documents:
            doc_id = self._get_document_id(doc)
            doc_hash = self._compute_document_hash(doc)
            hashes[doc_id] = doc_hash
        
        self.document_hashes_file.parent.mkdir(exist_ok=True)
        with open(self.document_hashes_file, 'w') as f:
            json.dump(hashes, f, indent=2)
        
        logger.debug(f"Saved hashes for {len(hashes)} documents")
    
    async def _load_document_hashes(self) -> Dict[str, str]:
        """
        Load document hashes from file
        
        Returns:
            Dictionary of document ID to hash
        """
        if not self.document_hashes_file.exists():
            return {}
        
        try:
            with open(self.document_hashes_file, 'r') as f:
                hashes = json.load(f)
            logger.debug(f"Loaded hashes for {len(hashes)} documents")
            return hashes
        except Exception as e:
            logger.warning(f"Failed to load document hashes: {e}")
            return {}
    
    async def get_index_stats(self) -> Dict:
        """
        Get Pinecone index statistics
        
        Returns:
            Dictionary with index stats
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            stats = self.pinecone_index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': Config.PINECONE_DIMENSION,
                'index_name': Config.PINECONE_INDEX_NAME,
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}


# Global service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create global embedding service instance
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
