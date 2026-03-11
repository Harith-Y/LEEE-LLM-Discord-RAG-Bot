"""
LEEE Discord RAG Bot - Main bot with improved architecture
Optimized for both local development and Render deployment
"""
from interactions import Client, Intents, slash_command, SlashContext, listen, slash_option, OptionType
from dotenv import load_dotenv
import os
import asyncio
from aiohttp import web
import logging
import sys

# Import new services and utilities
from src.config import Config
from src.utils.metrics import metrics
from src.utils.rate_limiter import get_rate_limiter
from src.services.querying import get_query_service

# Load environment variables
load_dotenv()

# Initialize logging - Render-compatible (console logging)
def setup_logging():
    """Setup logging that works for both local and Render"""
    logger = logging.getLogger()
    logger.handlers.clear()
    
    # Detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (works everywhere)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler only if not on Render (optional local logging)
    if not os.getenv('RENDER'):
        try:
            from pathlib import Path
            from logging.handlers import RotatingFileHandler
            
            log_dir = Path(Config.LOG_DIR)
            log_dir.mkdir(exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_dir / 'bot.log',
                maxBytes=10*1024*1024,
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Could not setup file logging: {e}")
    
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    
    # Suppress noisy loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()

# Initialize bot
bot = Client(intents=Intents.DEFAULT | Intents.GUILD_MESSAGES | Intents.MESSAGE_CONTENT)

# Initialize services
query_service = get_query_service()
rate_limiter = get_rate_limiter(
    max_requests=Config.RATE_LIMIT_MAX_REQUESTS,
    window_seconds=Config.RATE_LIMIT_WINDOW_SECONDS
)


# Health check server for Render/Fly.io
async def health_check(request):
    """Simple health check endpoint"""
    return web.Response(text="OK", status=200)


async def start_health_server():
    """Start the health check HTTP server"""
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', Config.PORT)
    await site.start()
    
    logger.info(f"Health check server running on 0.0.0.0:{Config.PORT}")


@listen()
async def on_ready():
    """Bot ready event handler"""
    platform = "Render ☁️" if os.getenv('RENDER') else "Local 💻"
    logger.info("=" * 60)
    logger.info(f"Bot Ready on {platform}")
    logger.info(f"Logged in as {bot.user}")
    
    # Synchronize slash commands
    logger.info("Syncing commands...")
    try:
        await bot.synchronise_interactions()
        logger.info("Commands synced successfully!")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}", exc_info=True)
    
    # Initialize query service
    try:
        logger.info("Initializing query service...")
        await query_service.initialize()
        logger.info("Query service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize query service: {e}", exc_info=True)
    
    logger.info("=" * 60)


@listen()
async def on_message_create(event):
    """Message creation event handler for logging"""
    if event.message.content and not event.message.author.bot:
        logger.debug(f"Message from {event.message.author}: {event.message.content[:100]}")


@slash_command(name="query", description="Ask a question about IIIT Hyderabad's LEEE program")
@slash_option(
    name="input_text",
    description="Your question about LEEE",
    required=True,
    opt_type=OptionType.STRING,
)
async def get_response(ctx: SlashContext, input_text: str):
    """
    Handle query command with rate limiting, validation, and error handling
    
    Args:
        ctx: Slash command context
        input_text: User's query
    """
    user_id = str(ctx.author.id)
    logger.info(f"Query from user {user_id} ({ctx.author}): {input_text[:100]}")
    
    # Defer response since processing might take time
    await ctx.defer()
    
    try:
        # Rate limiting check
        is_allowed, seconds_until_reset = await rate_limiter.is_allowed(user_id)
        
        if not is_allowed:
            error_msg = (
                f"⏱️ Rate limit exceeded! Please wait {seconds_until_reset} seconds before trying again.\n\n"
                f"You can make {Config.RATE_LIMIT_MAX_REQUESTS} requests every "
                f"{Config.RATE_LIMIT_WINDOW_SECONDS} seconds."
            )
            logger.warning(f"Rate limit exceeded for user {user_id}")
            await ctx.send(error_msg, ephemeral=True)
            return
        
        # Process query
        try:
            response = await query_service.process_query(input_text)
            
            # Format response with query
            full_response = f'**Query**: {input_text}\n\n{response}'
            
            # Split into multiple messages if needed (Discord limit: 2000 chars)
            if len(full_response) > 2000:
                chunks = []
                remaining_text = full_response
                CONTINUATION_PREFIX = '**...continued:**\n\n'
                PREFIX_LENGTH = len(CONTINUATION_PREFIX)
                
                # First chunk can be up to 1990 chars
                is_first = True
                
                while remaining_text:
                    # Adjust max length based on whether this will be a continuation
                    max_length = 1990 if is_first else (2000 - PREFIX_LENGTH - 10)  # 1970 for continuations
                    
                    # If remaining text fits, take it all
                    if len(remaining_text) <= max_length:
                        chunks.append(remaining_text)
                        break
                    
                    # Find a good break point (try newline, then space)
                    chunk = remaining_text[:max_length]
                    break_point = chunk.rfind('\n')
                    if break_point == -1:
                        break_point = chunk.rfind(' ')
                    if break_point == -1:
                        break_point = max_length  # Force break if no space found
                    
                    chunks.append(remaining_text[:break_point])
                    remaining_text = remaining_text[break_point:].lstrip()
                    is_first = False
                
                # Send chunks
                for i, chunk in enumerate(chunks):
                    content = chunk if i == 0 else f'{CONTINUATION_PREFIX}{chunk}'
                    try:
                        if i == 0:
                            await ctx.send(content)
                        else:
                            # Use channel.send for continuations to bypass interaction token limits
                            await ctx.channel.send(content)
                    except Exception as chunk_err:
                        logger.error(f"Failed to send chunk {i + 1}/{len(chunks)} for user {user_id}: {chunk_err}")
                        # Fallback: try ctx.send if channel.send failed, or vice versa
                        try:
                            await ctx.send(content)
                        except Exception:
                            logger.error(f"Fallback send also failed for chunk {i + 1}, stopping")
                            break
                
                logger.info(f"Successfully responded to user {user_id} (split into {len(chunks)} messages)")
            else:
                await ctx.send(full_response)
                logger.info(f"Successfully responded to user {user_id}")
            
        except ValueError as e:
            # Validation error
            error_msg = f"❌ Invalid query: {str(e)}\n\nPlease check your input and try again."
            logger.warning(f"Validation error for user {user_id}: {e}")
            await ctx.send(error_msg, ephemeral=True)
            
        except Exception as e:
            # Other errors
            error_msg = (
                "❌ Sorry, I encountered an error processing your query. "
                "Please try again or contact an administrator if the issue persists."
            )
            logger.error(f"Error processing query for user {user_id}: {e}", exc_info=True)
            await ctx.send(error_msg, ephemeral=True)
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.critical(f"Unexpected error in query handler: {e}", exc_info=True)
        try:
            await ctx.send(
                "❌ An unexpected error occurred. Please try again later.",
                ephemeral=True
            )
        except:
            pass  # Failed to send error message


@slash_command(name="stats", description="View bot statistics and health")
async def show_stats(ctx: SlashContext):
    """
    Show bot statistics
    
    Args:
        ctx: Slash command context
    """
    await ctx.defer(ephemeral=True)
    
    try:
        metrics_data = metrics.to_dict()
        query_stats = await query_service.get_stats()
        
        # Build stats message
        platform = "Render ☁️" if os.getenv('RENDER') else "Local 💻"
        stats_message = f"📊 **Bot Statistics** ({platform})\n\n"
        stats_message += f"**Queries:**\n"
        stats_message += f"• Total: {metrics_data['total_queries']}\n"
        stats_message += f"• Successful: {metrics_data['successful_queries']}\n"
        stats_message += f"• Failed: {metrics_data['failed_queries']}\n"
        stats_message += f"• Success Rate: {metrics_data['success_rate']}\n\n"
        
        stats_message += f"**Performance:**\n"
        stats_message += f"• Avg Response Time: {metrics_data['average_response_time']}\n"
        stats_message += f"• Cache Hit Rate: {metrics_data['cache_hit_rate']}\n"
        stats_message += f"• Uptime: {metrics_data['uptime_seconds']}\n\n"
        
        stats_message += f"**Cache:**\n"
        cache_stats = query_stats['cache']
        stats_message += f"• Response Cache: {cache_stats['response_cache_size']}/{cache_stats['response_cache_max_size']}\n"
        stats_message += f"• Index Cached: {'Yes' if cache_stats['index_cached'] else 'No'}\n\n"
        
        stats_message += f"**Index:**\n"
        index_stats = query_stats['index']
        
        if os.getenv('RENDER'):
            stats_message += "⚠️ Note: Stats reset on Render restarts"
        stats_message += f"• Total Vectors: {index_stats.get('total_vectors', 'N/A')}\n"
        
        await ctx.send(stats_message, ephemeral=True)
        
    except Exception as e:
        logger.error(f"Error showing stats: {e}", exc_info=True)
        await ctx.send("❌ Failed to retrieve statistics", ephemeral=True)


@slash_command(name="help", description="Get help on using the bot")
async def show_help(ctx: SlashContext):
    """
    Show help information
    """
    platform = "Render ☁️" if os.getenv('RENDER') else "Local 💻"
    help_message = f"""
    📖 **LEEE Bot Help** ({platform})
    
    **Available Commands:**
    • `/query <question>` - Ask a question about LEEE
    • `/stats` - View bot statistics
    • `/help` - Show this help message
    
    **Usage Tips:**
    • Be specific in your questions
    • Ask about LEEE-related topics only
    • Check #resources for comprehensive information
    
    **Rate Limits:**
    • {Config.RATE_LIMIT_MAX_REQUESTS} requests per {Config.RATE_LIMIT_WINDOW_SECONDS}s
    • Exceeded? Wait briefly and try again
    
    **Examples:**
    • "What subjects are covered in LEEE?"
    • "What is the LEEE syllabus?"
    • "How do I prepare for LEEE?"
    """
    
    await ctx.send(help_message, ephemeral=True)


async def main():
    """Main entry point"""
    try:
        platform = "Render" if os.getenv('RENDER') else "Local"
        logger.info(f"Starting LEEE Discord RAG Bot on {platform}...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Python path: {sys.executable}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Validate configuration
        try:
            Config.validate()
        except ValueError as e:
            logger.critical(f"Configuration validation failed: {e}")
            return
        
        # Start health check server and bot concurrently
        await asyncio.gather(
            start_health_server(),
            bot.astart(Config.DISCORD_BOT_TOKEN)
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Bot shutdown complete")
        # Log final metrics
        metrics.log_summary()


if __name__ == "__main__":
    asyncio.run(main())
