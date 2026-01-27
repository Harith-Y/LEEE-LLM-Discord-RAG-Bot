from llama_index.core.settings import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core import StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging
import sys
import os
import time


load_dotenv()

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Configure LLM: OpenRouter with best free model
# Always use NVIDIA NIM embeddings
embed_model = NVIDIAEmbedding(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=os.getenv("NVIDIA_API_KEY"),
    truncate="END"
)

logging.info("Initializing OpenRouter with meta-llama/llama-3.3-70b-instruct:free...")
llm = OpenRouter(
    model="meta-llama/llama-3.3-70b-instruct:free",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
logging.info("Successfully initialized OpenRouter LLM with NVIDIA NIM embeddings")

# Configure global settings (replaces ServiceContext in newer versions)
Settings.llm = llm
Settings.embed_model = embed_model

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "leee-helpbot-nvidia-embeddings"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    logging.info(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1024,  # NVIDIA nv-embedqa-e5-v5 dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    logging.info(f"Pinecone index '{index_name}' created successfully.")
else:
    logging.info(f"Pinecone index '{index_name}' already exists.")

pinecone_index = pc.Index(index_name)


async def load_index(directory_path: str = r'data'):
    """Load or create index from Pinecone (embeddings generated only once)"""
    documents = SimpleDirectoryReader(directory_path).load_data()
    logging.info(f"Loaded documents with {len(documents)} pages")
    
    # Check if index has data
    stats = pinecone_index.describe_index_stats()
    
    if stats['total_vector_count'] > 0:
        # Load existing embeddings from Pinecone
        logging.info(f"Loading existing index from Pinecone ({stats['total_vector_count']} vectors)...")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store)
        logging.info("Index loaded from Pinecone (no re-embedding needed).")
    else:
        # Create and store new embeddings in Pinecone
        logging.info("No existing embeddings found. Creating new index and storing in Pinecone...")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        logging.info("New index created and stored in Pinecone.")
    
    return index

async def update_index(directory_path: str = r'data'):
    """Update index in Pinecone by clearing and re-embedding all documents"""
    try:
        documents = SimpleDirectoryReader(directory_path).load_data()
        logging.info(f"Loaded {len(documents)} documents for update")
    except FileNotFoundError:
        logging.error("Invalid document directory path.")
        return None
    
    try:
        # Clear all existing vectors
        logging.info("Clearing existing vectors from Pinecone...")
        stats = pinecone_index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        
        if not namespaces:
            namespaces = ['']  # Default namespace
        
        for ns in namespaces:
            pinecone_index.delete(delete_all=True, namespace=ns)
        
        logging.info("Existing vectors cleared. Creating fresh embeddings...")
        
        # Create fresh embeddings
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        # Verify upload
        stats = pinecone_index.describe_index_stats()
        logging.info(f"Index updated successfully. Total vectors: {stats['total_vector_count']}")
        return len(documents)
        
    except Exception as e:
        logging.error(f"Error updating index: {e}")
        return None

