from typing import List, Optional
from app.services.qdrant_client import QdrantClient
from app.services.embedder import EmbedderService
from app.config import settings
from app.models.schema import SearchResult

class RetrieverService:
    def __init__(self):
        self.qdrant = QdrantClient()
        self.embedder = EmbedderService()

    async def retrieve_similar_chunks(
        self,
        text: str,
        top_k: int = None,
        reference_only: bool = True,
        contract_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        clause_type: Optional[str] = None,
        min_similarity: float = None,
    ) -> List[SearchResult]:
        k = top_k or settings.retrieval_top_k
        thr = min_similarity or settings.min_similarity

        q_emb = await self.embedder.create_embedding(text)
        qf = self.qdrant.build_filter(
            reference_only=reference_only,
            contract_type=contract_type,
            jurisdiction=jurisdiction,
            clause_type=clause_type,
        )
        # 후보를 넉넉히 가져온 뒤 임계값 컷
        candidates = await self.qdrant.search_similar(q_emb, limit=k * 2, query_filter=qf)
        filtered = [c for c in candidates if c.similarity_score >= thr]
        return filtered[:k]
