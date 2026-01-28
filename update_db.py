"""
Standalone script to update the database from the data folder
Run this script to re-embed and update the Pinecone vector database with documents from the data folder
"""
import asyncio
from manage_embedding import update_index
import logging
import sys

# Configure logging
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    """Main function to update the database"""
    print("=" * 60)
    print("DATABASE UPDATE SCRIPT")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Clear all existing vectors from Pinecone")
    print("2. Load all documents from the 'data' folder")
    print("3. Create new embeddings and upload to Pinecone")
    print("\n" + "=" * 60)
    
    # Confirm before proceeding
    confirm = input("\nProceed with database update? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Update cancelled.")
        return
    
    print("\nStarting database update...")
    print("-" * 60)
    
    # Update the index
    result = await update_index()
    
    print("-" * 60)
    if result:
        print(f"\n✓ SUCCESS: Database updated with {result} documents")
    else:
        print("\n✗ ERROR: Failed to update database")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
