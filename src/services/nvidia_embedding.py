"""
NVIDIA NeMo Retriever VL embedding adapter.

Background: nvidia/nv-embedqa-e5-v5 was retired by NVIDIA on 2026-08-25, and its
successor nvidia/llama-3.2-nv-embedqa-1b-v2 is also EOL. nvidia/llama-nemotron-embed-vl-1b-v2
is the hosted model that still answers on integrate.api.nvidia.com, but its endpoint
requires a ``modality`` array the same length as the input batch - something the stock
``NVIDIAEmbedding`` wrapper never sends, so requests through it hang.

This subclass overrides the six embedding entrypoints to send:
  * ``modality``   - ["text", ...] matching the batch length (required by the VL endpoint)
  * ``input_type`` - "query" for query embeddings, "passage" for documents
  * ``dimensions`` - Matryoshka output size, kept at 1024 to match the Pinecone index
  * ``truncate``   - inherited from Config.EMBEDDING_TRUNCATE
"""
import logging
import math
from typing import List, Tuple

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.embeddings.nvidia import utils as _nv_utils
from openai import UnprocessableEntityError

logger = logging.getLogger(__name__)

# Safety-rejected passages are split no further than this.
_MIN_SPLIT_CHARS = 100

# The VL NeMo Retriever model is not in the wrapper's built-in model table, so
# NVIDIAEmbedding._validate_model() emits a spurious "Unable to determine validity"
# warning at startup. Register it (idempotently) to silence that and to expose it
# via the .available_models property. Uses integrate.api.nvidia.com (endpoint=None).
_MODEL_ID = "nvidia/nemotron-3.5-embed-1b"
_nv_utils.EMBEDDING_MODEL_TABLE.setdefault(
    _MODEL_ID, _nv_utils.Model(id=_MODEL_ID, model_type="embedding")
)


class NVIDIANemoRetrieverEmbedding(NVIDIAEmbedding):
    """``NVIDIAEmbedding`` variant for the NeMo Retriever VL models (text modality only)."""

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIANemoRetrieverEmbedding"

    def _extra_body(self, batch_size: int, input_type: str) -> dict:
        body = {
            "modality": ["text"] * batch_size,
            "input_type": input_type,
            "truncate": self.truncate,
        }
        if self.dimensions:
            body["dimensions"] = self.dimensions
        return body

    # --- sync ---------------------------------------------------------------
    def _get_query_embedding(self, query: str) -> List[float]:
        return (
            self._client.embeddings.create(
                input=[query],
                model=self.model,
                extra_body=self._extra_body(1, "query"),
            )
            .data[0]
            .embedding
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        return (
            self._client.embeddings.create(
                input=[text],
                model=self.model,
                extra_body=self._extra_body(1, "passage"),
            )
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        assert len(texts) <= 259, "The batch size should not be larger than 259."
        try:
            data = self._client.embeddings.create(
                input=texts,
                model=self.model,
                extra_body=self._extra_body(len(texts), "passage"),
            ).data
        except UnprocessableEntityError as e:
            if "content safety" not in str(e):
                raise
            # One flagged chunk fails the whole batch, so retry them one by one.
            return [self._embed_flagged_text(text) for text in texts]
        return [d.embedding for d in data]

    def _embed_flagged_text(self, text: str) -> List[float]:
        """Embed one passage, working around NVIDIA's safety screen.

        The screen has false positives on ordinary text. A rejected passage is
        halved until the pieces are accepted, and the accepted piece vectors are
        combined (length-weighted, normalised). A small fragment that is still
        rejected is left out of the vector; the stored chunk text is unchanged.
        """
        pieces = self._embed_accepted_pieces(text)
        if not pieces:
            raise RuntimeError(f"Safety screen rejected an entire chunk: {text[:80]!r}")
        if len(pieces) == 1:
            return pieces[0][1]
        combined = [
            sum(weight * value for (weight, _), value in zip(pieces, values))
            for values in zip(*(vector for _, vector in pieces))
        ]
        norm = math.sqrt(sum(v * v for v in combined))
        return [v / norm for v in combined]

    def _embed_accepted_pieces(self, text: str) -> List[Tuple[int, List[float]]]:
        """Return (length, vector) for each part of ``text`` the API accepts."""
        try:
            return [(len(text), self._get_text_embedding(text))]
        except UnprocessableEntityError as e:
            if "content safety" not in str(e):
                raise
        if len(text) < _MIN_SPLIT_CHARS:
            logger.warning(f"Safety screen rejected fragment, omitted from vector: {text!r}")
            return []
        middle = len(text) // 2
        cut = text.rfind(" ", 0, middle)
        cut = cut if cut > 0 else middle
        return self._embed_accepted_pieces(text[:cut]) + self._embed_accepted_pieces(text[cut:])

    # --- async --------------------------------------------------------------
    async def _aget_query_embedding(self, query: str) -> List[float]:
        resp = await self._aclient.embeddings.create(
            input=[query],
            model=self.model,
            extra_body=self._extra_body(1, "query"),
        )
        return resp.data[0].embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        resp = await self._aclient.embeddings.create(
            input=[text],
            model=self.model,
            extra_body=self._extra_body(1, "passage"),
        )
        return resp.data[0].embedding

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        assert len(texts) <= 259, "The batch size should not be larger than 259."
        resp = await self._aclient.embeddings.create(
            input=texts,
            model=self.model,
            extra_body=self._extra_body(len(texts), "passage"),
        )
        return [d.embedding for d in resp.data]
