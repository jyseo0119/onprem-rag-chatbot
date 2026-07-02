"""BGE-m3 embedder.

Runs locally in both the demo and on-prem deployments — so retrieval behaviour
is identical regardless of which LLM answers. Multilingual + strong on
Korean/technical text, which is why it's chosen over smaller API embeddings.

We use only BGE-m3's **dense** output here (1024-dim, cosine). The model also
emits sparse and ColBERT vectors; those are a deliberate stage-2+ option, not
wired into the demo retrieval path yet.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# BGE-m3 dense vector size. Asserted against the model at load time.
BGE_M3_DIM = 1024


class BGEM3Embedder:
    """Thin wrapper over FlagEmbedding's BGEM3FlagModel.

    The model (~2.3 GB) is downloaded/loaded lazily on first use so importing
    this module — or starting the API — stays cheap.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 8192,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            logger.info("loading %s on %s (first load downloads the model)",
                        self.model_name, self.device)
            self._model = BGEM3FlagModel(
                self.model_name,
                # fp16 only helps on GPU; on CPU it is slower/unsupported.
                use_fp16=self.device != "cpu",
                devices=self.device,
                normalize_embeddings=True,  # unit vectors -> cosine == dot product
            )
        return self._model

    @property
    def dimension(self) -> int:
        return BGE_M3_DIM

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        out = model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]  # numpy (n, 1024)
        return dense.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts for indexing."""
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (stage 2 retrieval)."""
        return self._encode([text])[0]
