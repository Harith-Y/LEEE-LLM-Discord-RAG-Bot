# LEEE LLM Discord RAG Bot

A Discord bot that uses Retrieval-Augmented Generation (RAG) to answer questions about LEEE (Lateral Entry Exam for Engineers) using LlamaIndex, Pinecone vector database, and various LLM providers (NVIDIA, OpenRouter). The bot provides accurate, context-aware responses by retrieving relevant information from a curated knowledge base.

<img src="demo/demo_query.gif">

## Features

- 🤖 **RAG-powered responses** using LlamaIndex and Pinecone
- 🔄 **Dynamic database updates** via Discord commands
- 🚀 **Multiple LLM support** (NVIDIA NIM, OpenRouter)
- 📚 **Custom knowledge base** from markdown and text documents
- ☁️ **Cloud deployment ready** (Render)

## Prerequisites

- Python 3.10 or higher
- Discord Bot Token ([Discord Developer Portal](https://discord.com/developers/applications))
- NVIDIA API Key ([NVIDIA NIM](https://build.nvidia.com/))
- OpenRouter API Key ([OpenRouter](https://openrouter.ai/))
- Pinecone API Key ([Pinecone](https://www.pinecone.io/))

## Local Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/LEEE-LLM-Discord-RAG-Bot.git
cd LEEE-LLM-Discord-RAG-Bot
```

2. Create and activate a virtual environment:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Unix/MacOS:**
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

Rename `.env.example` to `.env` and add your API keys:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token_here
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

5. Add your documents:

Place your `.txt` and `.md` files in the `data/` folder.

6. Run the bot:

**Windows:**
```bash
run.bat
```

**Unix/MacOS:**
```bash
python bot.py
```

## Getting Started

1. Clone the repository to your local machine:

```bash
git clone https://github.com/nur-zaman/LLM-RAG-Bot.git
cd LLM-RAG-Bot
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

3. Activate the virtual environment:

- On Windows:

```bash
venv\Scripts\activate
```

- On Unix or MacOS:

```bash
source venv/bin/activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Rename the `.env.example` file to `.env` in the project root directory and add your tokens and api keys:

```env
DISCORD_BOT_TOKEN=your_bot_token_here
OPENAI_API_KEY=your_openai_api_key
```

## Running the Bot

Execute the `run.bat` file to start the bot:

```bash
run.bat
```

This will launch the bot, and you should see "Ready" in the console once it has successfully connected to Discord.

## Bot Commands

### `/query`

Query the LEEE knowledge base with your question.

**Usage:**
```
/query input_text: What is LEEE?
```

**Parameters:**
- `input_text` (required): Your question or query

**Example:**
```
/query input_text: What are the eligibility criteria for LEEE?
```

### `/updatedb`

Updates the knowledge base with new documents from the `data/` folder.

**Usage:**
```
/updatedb
```

This command will scan the `data/` folder and index any new or modified documents into the Pinecone vector database.

## Project Structure

```
LEEE-LLM-Discord-RAG-Bot/
├── bot.py                  # Main Discord bot file
├── querying.py            # RAG query logic
├── manage_embedding.py    # Document indexing and embedding
├── clear_pinecone.py     # Utility to clear Pinecone index
├── requirements.txt       # Python dependencies
├── Procfile              # Render worker configuration
├── render.yaml           # Render Blueprint configuration
├── .env.example          # Environment variables template
├── data/                 # Knowledge base documents
│   ├── Bot Intro.txt
│   ├── LEEE Index.md
│   └── ... (your documents)
└── demo/                 # Demo assets
```

## Bot Usage

The bot responds to a single slash command:

### `/query`

- **Description:** Enter your query :)
- **Options:**
  - `input_text` (required): The input text for the query.

### `/updatedb`

- **Description:** Updates your information database
## Technologies Used

- **[discord-py-interactions](https://github.com/interactions-py/interactions.py)** - Discord bot framework
- **[LlamaIndex](https://www.llamaindex.ai/)** - RAG framework for document indexing and querying
- **[Pinecone](https://www.pinecone.io/)** - Vector database for semantic search
- **[NVIDIA NIM](https://build.nvidia.com/)** - LLM embeddings
- **[OpenRouter](https://openrouter.ai/)** - LLM inference API

## Troubleshooting

### Bot not responding
- Verify your `DISCORD_BOT_TOKEN` is correct
- Ensure the bot has proper permissions in your Discord server
- Check that slash commands are synced (may take up to 1 hour)

### Query errors
- Verify all API keys are set correctly
- Check Pinecone index exists and has data
- Run `/updatedb` to ensure documents are indexed

### Deployment issues
- Ensure all environment variables are set in Render
- Check Render logs for error messages
- Verify `requirements.txt` has all dependencies
## Additional Notes

- [llamaIndex documentation](https://docs.llamaindex.ai/en/stable/).
- [refreshing-private-data-sources-with-llamaindex](https://betterprogramming.pub/refreshing-private-data-sources-with-llamaindex-document-management-1d1f1529f5eb).
