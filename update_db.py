"""
Update database script using the new architecture
Run this script to re-embed and update the Pinecone vector database
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.utils.logging_config import setup_logging
from src.services.embedding import get_embedding_service
from src.utils.cache import get_cache_manager

# Setup logging
logger = setup_logging(log_level='INFO', log_dir=Config.LOG_DIR)


async def main():
    """Main function to update the database"""
    print("=" * 70)
    print(" " * 20 + "DATABASE UPDATE SCRIPT")
    print("=" * 70)
    print()
    print("This script will update your Pinecone vector database.")
    print()
    print("Choose update mode:")
    print("  1. Incremental (recommended) - Only update changed/new documents")
    print("  2. Full rebuild - Clear all and re-embed everything")
    print()
    
    # Get user choice
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    incremental = (choice == '1')
    mode_name = "Incremental" if incremental else "Full Rebuild"
    
    print()
    print("=" * 70)
    print(f"Update Mode: {mode_name}")
    print(f"Data Directory: {Config.DATA_DIR}")
    print("=" * 70)
    
    if not incremental:
        print()
        print("⚠️  WARNING: Full rebuild will:")
        print("   - Clear all existing vectors from Pinecone")
        print("   - Re-embed all documents (may take time)")
        print("   - Invalidate all cached responses")
        print()
    
    # Confirm before proceeding
    confirm = input(f"\nProceed with {mode_name} update? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("\nUpdate cancelled.")
        return
    
    print()
    print("-" * 70)
    print("Starting database update...")
    print("-" * 70)
    
    try:
        # Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = get_embedding_service()
        await embedding_service.initialize()
        logger.info("Embedding service initialized")
        
        # Update the index
        logger.info(f"Starting {mode_name} update...")
        result = await embedding_service.update_index(
            directory_path=Config.DATA_DIR,
            incremental=incremental
        )
        
        print("-" * 70)
        
        if result is not None and result > 0:
            print()
            print(f"✓ SUCCESS: Updated {result} documents")
            print()
            
            # Clear response cache since documents changed
            cache_manager = get_cache_manager()
            await cache_manager.clear_all_caches()
            print("✓ Response cache cleared")
            
            # Show index stats
            stats = await embedding_service.get_index_stats()
            print()
            print("Index Statistics:")
            print(f"  - Total Vectors: {stats.get('total_vectors', 'N/A')}")
            print(f"  - Index Name: {stats.get('index_name', 'N/A')}")
            print(f"  - Dimension: {stats.get('dimension', 'N/A')}")
            
        elif result == 0:
            print()
            print("ℹ️  No changes detected. Database is already up to date.")
            print()
        else:
            print()
            print("✗ ERROR: Failed to update database")
            print("   Check logs for details")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error during database update: {e}", exc_info=True)
        print()
        print(f"✗ ERROR: {e}")
        print("   Check logs for details")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("Database update complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nUpdate cancelled by user.")
        sys.exit(0)
