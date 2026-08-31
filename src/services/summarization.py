"""
Channel summarization service.

Produces a concise summary of recent Discord messages using the same
multi-provider LLM cascade the query service uses. No retrieval / vector store
is involved - this is a pure summarization pass over the supplied transcript.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

from llama_index.core.settings import Settings

from src.services.embedding import get_embedding_service
from src.utils.metrics import MetricsContext
from src.utils.discord_formatter import format_for_discord

logger = logging.getLogger(__name__)

# Keep the transcript comfortably under free-tier context limits.
MAX_TRANSCRIPT_CHARS = 14000
# Per-message hard cap so one huge paste can't crowd out the rest of the chat.
MAX_MESSAGE_CHARS = 600
LLM_TIMEOUT = 90  # seconds - free-tier models are slow


SUMMARY_TEMPLATE = (
    "You are summarizing a Discord chat conversation for someone who missed it.\n\n"
    "**Instructions:**\n"
    "1. Write a concise summary of the conversation below.\n"
    "2. Start with 2-4 sentences of overall context, then use bullet points for the key "
    "topics discussed, questions asked, answers or decisions reached, and any action "
    "items or links shared.\n"
    "3. Attribute important points to the person who made them (use their display name).\n"
    "4. Preserve any URLs exactly as written.\n"
    "5. Ignore bot messages, slash-command invocations, and pure noise (bare greetings, "
    "single-word reactions).\n"
    "6. Do NOT invent information that is not in the transcript. If the transcript is too "
    "sparse to summarize meaningfully, say so plainly.\n"
    "7. Use plain Markdown only - no tables, no HTML.\n\n"
    "**Conversation ({message_count} messages, oldest first):**\n"
    "---------------------\n"
    "{transcript}\n"
    "---------------------\n\n"
    "**Summary:**"
)


@dataclass
class ChatMessage:
    """Minimal representation of a Discord message for summarization."""

    author: str
    content: str
    timestamp: str  # pre-formatted, e.g. "14:03"


class SummarizationService:
    """Summarize a list of chat messages via the shared LLM cascade."""

    def __init__(self) -> None:
        self.embedding_service = get_embedding_service()

    async def initialize(self) -> None:
        # The LLM cascade is built by the embedding service; ensure it exists.
        await self.embedding_service.initialize()

    def build_transcript(self, messages: List[ChatMessage]) -> str:
        """Render messages oldest->newest, capping per-message and total length."""
        lines: List[str] = []
        for m in messages:
            content = m.content.strip()
            if not content:
                continue
            if len(content) > MAX_MESSAGE_CHARS:
                content = content[:MAX_MESSAGE_CHARS] + " […]"
            lines.append(f"[{m.timestamp}] {m.author}: {content}")

        transcript = "\n".join(lines)
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            # Keep the most recent content; drop from the top.
            transcript = (
                "… (earlier messages trimmed) …\n"
                + transcript[-MAX_TRANSCRIPT_CHARS:]
            )
        return transcript

    async def summarize(self, messages: List[ChatMessage]) -> str:
        """
        Summarize the given messages.

        Args:
            messages: Chat messages in chronological (oldest-first) order.

        Returns:
            Discord-friendly summary text.

        Raises:
            Exception: If every model in the LLM cascade fails.
        """
        await self.initialize()

        transcript = self.build_transcript(messages)
        if not transcript.strip():
            return (
                "There's nothing to summarize — the recent messages have no "
                "readable text content."
            )

        prompt = SUMMARY_TEMPLATE.format(
            message_count=len(messages), transcript=transcript
        )

        llm_chain = self.embedding_service.get_llm_chain()
        original_llm = Settings.llm
        response_text: Optional[str] = None
        last_error: Optional[Exception] = None

        with MetricsContext("summarize_llm"):
            for provider, model_name, llm in llm_chain:
                try:
                    logger.info(f"Summarize: trying {provider}:{model_name}...")
                    Settings.llm = llm
                    response = await asyncio.wait_for(
                        llm.acomplete(prompt), timeout=LLM_TIMEOUT
                    )
                    response_text = response.text
                    logger.info(
                        f"Summarize: {provider}:{model_name} responded "
                        f"({len(response_text)} chars)"
                    )
                    break
                except Exception as e:  # cascade to the next model
                    last_error = e
                    logger.warning(
                        f"Summarize: {provider}:{model_name} failed: {e}"
                    )
                    continue
                finally:
                    Settings.llm = original_llm

        if response_text is None:
            logger.error(
                f"Summarize: all {len(llm_chain)} models failed. "
                f"Last error: {last_error}"
            )
            raise Exception(
                "All LLM providers failed or rate limited. Please try again later."
            )

        return format_for_discord(response_text)


# Global service instance
_summarization_service: Optional[SummarizationService] = None


def get_summarization_service() -> SummarizationService:
    """Get or create the global summarization service instance."""
    global _summarization_service
    if _summarization_service is None:
        _summarization_service = SummarizationService()
    return _summarization_service
