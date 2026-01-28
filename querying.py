from manage_embedding import load_index
from llama_index.core.settings import Settings
from llama_index.core.prompts import PromptTemplate
import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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


async def data_querying(input_text: str):
    # Load index
    index = await load_index("data")
    
    # Get retriever and retrieve relevant chunks
    retriever = index.as_retriever(similarity_top_k=10)
    nodes = await retriever.aretrieve(input_text)
    
    # Extract all retrieved text
    retrieved_text = "\n\n---\n\n".join([node.text for node in nodes])
    
    # Create prompt with all retrieved content
    prompt = f"""You are a helpful assistant answering questions about IIIT Hyderabad's Lateral Entry Exam (LEEE) and related academic programs.

Retrieved Information:
{retrieved_text}

User Question: {input_text}

Instructions:
- First, check if the question is relevant to LEEE, IIIT Hyderabad, or related academic topics
- If the question contains inappropriate language, curse words, or is completely unrelated to LEEE/IIITH, respond politely: "I'm designed to answer questions about IIIT Hyderabad's LEEE program. Please check the #resources channel for comprehensive information."
- If relevant, use ONLY the information from the retrieved content above
- Answer the question directly and concisely
- List all relevant topics/subjects mentioned in the retrieved information
- Include ALL information that answers the user's question
- Do NOT add information not present in the retrieved content
- If the retrieved content doesn't contain the answer, say: "I don't have specific information about this. Please check the #resources channel for comprehensive LEEE information."

Answer:"""
    
    # Try query with current LLM
    try:
        response = await Settings.llm.acomplete(prompt)
        response_text = response.text
        logging.info(response_text)
        return response_text
    except Exception as e:
        logging.error(f"Error during query: {e}")
        raise
