import asyncio
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient as QClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings
from app.models.schema import ClauseNormalized, SearchResult
from app.utils.logger import logger

class QdrantClient:
    def __init__(self):
        self.client = QClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection_name
        self.vec_dim = settings.embedding_dim
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            cols = self.client.get_collections().collections
            names = [c.name for c in cols]
            if self.collection not in names:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.vec_dim, distance=Distance.COSINE),
                )
                logger.info(f"Qdrant collection created: {self.collection}")
        except Exception as e:
            raise RuntimeError(f"Qdrant ensure_collection failed: {e}")

    async def upsert_clauses(
        self,
        contract_id: str,
        clauses: List[ClauseNormalized],
        embeddings: List[List[float]],
        meta: Dict[str, Any],
    ):
        def _upsert():
            points: List[PointStruct] = []
            for cl, emb in zip(clauses, embeddings):
                pid = str(uuid.uuid4())
                payload = {
                    "contract_id": contract_id,
                    "clause_id": cl.clause_id,
                    "content": cl.analysis_text,
                    "original_identifier": cl.original_identifier,
                    "start_index": cl.start_index,
                    "end_index": cl.end_index,
                    "clause_type": cl.clause_type,
                    "article_number": cl.article_number,
                    "paragraph_number": cl.paragraph_number,
                    # 검색 필터용 평평한 메타:
                    "contract_type": meta.get("contract_type"),
                    "jurisdiction": meta.get("jurisdiction"),
                    "language": meta.get("language"),
                    "issuer": meta.get("issuer"),
                    "version": meta.get("version"),
                    "source_uri": meta.get("source_uri"),
                    "is_reference": meta.get("is_reference", False),
                }
                points.append(PointStruct(id=pid, vector=emb, payload=payload))
            self.client.upsert(collection_name=self.collection, points=points)

        await asyncio.to_thread(_upsert)
        logger.info(f"Upserted {len(clauses)} points for contract={contract_id}")

    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        query_filter: Optional[Filter] = None,
    ) -> List[SearchResult]:
        def _search():
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            )
            return hits

        hits = await asyncio.to_thread(_search)
        results: List[SearchResult] = []
        for h in hits:
            payload = h.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(h.id),
                    content=payload.get("content", ""),
                    similarity_score=float(h.score),
                    metadata=payload,
                )
            )
        return results

    def build_filter(
        self,
        reference_only: bool = True,
        contract_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        clause_type: Optional[str] = None,
    ) -> Optional[Filter]:
        must = []
        if reference_only:
            must.append(FieldCondition(key="is_reference", match=MatchValue(value=True)))
        if contract_type:
            must.append(FieldCondition(key="contract_type", match=MatchValue(value=contract_type)))
        if jurisdiction:
            must.append(FieldCondition(key="jurisdiction", match=MatchValue(value=jurisdiction)))
        if clause_type:
            must.append(FieldCondition(key="clause_type", match=MatchValue(value=clause_type)))
        return Filter(must=must) if must else None
