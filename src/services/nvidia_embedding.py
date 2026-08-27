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
from typing import List

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.embeddings.nvidia import utils as _nv_utils

# The VL NeMo Retriever model is not in the wrapper's built-in model table, so
# NVIDIAEmbedding._validate_model() emits a spurious "Unable to determine validity"
# warning at startup. Register it (idempotently) to silence that and to expose it
# via the .available_models property. Uses integrate.api.nvidia.com (endpoint=None).
_MODEL_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"
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
        data = self._client.embeddings.create(
            input=texts,
            model=self.model,
            extra_body=self._extra_body(len(texts), "passage"),
        ).data
        return [d.embedding for d in data]

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
