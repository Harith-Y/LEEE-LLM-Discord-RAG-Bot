from interactions import Client, Intents, slash_command, SlashContext, listen,slash_option,OptionType
from dotenv import load_dotenv
import os
import asyncio
from aiohttp import web

from querying import data_querying
from manage_embedding import update_index

load_dotenv()


bot = Client(intents=Intents.DEFAULT | Intents.GUILD_MESSAGES | Intents.MESSAGE_CONTENT)

# Health check server for Render
async def health_check(request):
    return web.Response(text="Bot is running!", status=200)

async def start_health_server():
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv('PORT', 8080))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    print(f"Health check server running on port {port}")


@listen() 
async def on_ready():
    print("Ready")
    # print(f"This bot is owned by {bot.owner}")


@listen()
async def on_message_create(event):
    # This event is called when a message is sent in a channel the bot can see
    if event.message.content:
        print(f"message received: {event.message.content}")


@slash_command(name="query", description="Enter your query :)")
@slash_option(
    name="input_text",
    description="input text",
    required=True,
    opt_type=OptionType.STRING,
)
async def get_response(ctx: SlashContext, input_text: str):
    await ctx.defer()
    response = await data_querying(input_text)
    
    # Format response with query
    full_response = f'**Input Query**: {input_text}\n\n{response}'
    
    # Truncate if exceeds Discord's 2000 character limit
    if len(full_response) > 2000:
        max_response_length = 2000 - len(f'**Input Query**: {input_text}\n\n') - 50  # Reserve space for truncation message
        truncated = response[:max_response_length]
        full_response = f'**Input Query**: {input_text}\n\n{truncated}\n\n...[Response truncated]'
    
    await ctx.send(full_response)

@slash_command(name="updatedb", description="Update your information database :)")
async def updated_database(ctx: SlashContext):
    await ctx.defer()
    update = await update_index()
    if update:
        response = f'Successfully updated database with {update} documents'
    else:
        response = f'Error updating index'
    await ctx.send(response)


async def main():
    # Start health check server
    await start_health_server()
    # Start bot
    await bot.astart(os.getenv("DISCORD_BOT_TOKEN"))

if __name__ == "__main__":
    asyncio.run(main())