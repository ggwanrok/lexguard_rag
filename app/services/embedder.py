from typing import List
import numpy as np
from openai import AsyncOpenAI
from app.config import settings

class EmbedderService:
    """OpenAI Embedding (v1, Async)"""

    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되어야 합니다.")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.batch_size = 256

    async def create_embedding(self, text: str) -> List[float]:
        resp = await self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        embs: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = await self.client.embeddings.create(model=self.model, input=batch)
            embs.extend([d.embedding for d in resp.data])
        return embs

    def cosine_sim(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)
