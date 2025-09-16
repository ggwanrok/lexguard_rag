from typing import List, Literal, Optional
import numpy as np
import asyncio
from app.config import settings

# OpenAI (async)
from openai import AsyncOpenAI

# Hugging Face (local, sync -> thread offload)
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # optional runtime import
    SentenceTransformer = None  # type: ignore


Provider = Literal["openai", "huggingface"]


class EmbedderService:
    """
    Embedding provider abstraction.
    - provider = "huggingface": SentenceTransformer(local) ex) kakao1513/KURE-legal-ft-v1
    - provider = "openai": OpenAI embeddings API (existing path)

    expose:
      - create_embedding(text) -> List[float]
      - create_embeddings(texts) -> List[List[float]]
      - cosine_sim(v1, v2) -> float
      - dim (embedding dimension)
    """

    def __init__(self, provider: Provider | None = None):
        self.provider: Provider = provider or settings.embedding_provider
        self.batch_size = 256
        self.dim: Optional[int] = None

        if self.provider == "huggingface":
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers가 설치되어 있지 않습니다. "
                    "requirements.txt를 업데이트해 주세요."
                )
            model_name = settings.hf_model_name
            device = settings.hf_device  # "cpu" | "cuda" | "mps"
            self.hf_model = SentenceTransformer(model_name, device=device)
            # 임베딩 차원 자동 인식
            try:
                self.dim = int(self.hf_model.get_sentence_embedding_dimension())
            except Exception:
                # fallback: 더미 인코딩
                v = self.hf_model.encode(["dim probe"], normalize_embeddings=True)
                self.dim = int(v.shape[-1])
            # OpenAI 경로 미사용
            self.client = None
            self.model = None

        else:  # "openai"
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY가 설정되어야 합니다.")
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            self.model = settings.embedding_model
            # OpenAI 차원은 설정값 사용(기본 1536). 필요시 .env로 조정
            self.dim = settings.embedding_dim
            self.hf_model = None

    # -------------------
    # Embedding methods
    # -------------------
    async def create_embedding(self, text: str) -> List[float]:
        if self.provider == "huggingface":
            # CPU/GPU 동기 연산 → 스레드 오프로딩
            def _encode_one() -> List[float]:
                vec = self.hf_model.encode([text], normalize_embeddings=True)  # (1, D)
                return vec[0].tolist()
            return await asyncio.to_thread(_encode_one)

        # OpenAI
        resp = await self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "huggingface":
            def _encode_many() -> List[List[float]]:
                vecs = self.hf_model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
                return vecs.tolist()
            return await asyncio.to_thread(_encode_many)

        # OpenAI
        embs: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            resp = await self.client.embeddings.create(model=self.model, input=batch)
            embs.extend([d.embedding for d in resp.data])
        return embs

    # -------------------
    # Utils
    # -------------------
    def cosine_sim(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)
