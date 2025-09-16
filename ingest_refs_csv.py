#!/usr/bin/env python3
"""
CSV의 레퍼런스 스니펫들을 OpenAI 임베딩 → Qdrant에 upsert 합니다.
- 프로젝트의 EmbedderService / QdrantClient / ClauseNormalized 스키마를 그대로 사용합니다.
- CSV 스키마(컬럼):
  text, category, subcategory, role_applicability, contract_type, jurisdiction, clause_type,
  identifier, title, source, version, is_reference, quality, tags, recommended_language, notes
"""

#!/usr/bin/env python3
import asyncio
import uuid
import sys
from typing import List
import pandas as pd

from app.services.embedder import EmbedderService
from app.services.qdrant_client import QdrantClient
from app.models.schema import ClauseNormalized

REQUIRED_COLS = ["text"]

def _ensure_cols(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

def _row_to_clause_and_meta(row: pd.Series):
    text = str(row.get("text", "") or "").strip()
    if not text:
        return None, None

    identifier = str(row.get("identifier") or row.get("title") or row.get("category") or "ref").strip()
    title = str(row.get("title") or "").strip()
    clause_type = str(row.get("clause_type") or "article").strip()

    clause = ClauseNormalized(
        clause_id=f"ref_{uuid.uuid4().hex[:8]}",
        original_identifier=identifier,
        original_text=text,
        analysis_text=text,
        start_index=0,
        end_index=len(text),
        clause_type=clause_type,
        article_number=None,
        paragraph_number=None,
        title=title or None,
    )

    meta = {
        "contract_id": f"ref_{uuid.uuid4().hex[:12]}",
        "contract_name": str(row.get("source") or "Reference"),
        "contract_type": str(row.get("contract_type") or "NDA"),
        "issuer": str(row.get("source") or "ReferenceLib"),
        "jurisdiction": str(row.get("jurisdiction") or "KR"),
        "language": "ko",
        "subject": "NDA",
        "version": str(row.get("version") or ""),
        "source_uri": None,
        "is_reference": bool(row.get("is_reference", True)),
        "category": str(row.get("category") or ""),
        "subcategory": str(row.get("subcategory") or ""),
        "role_applicability": str(row.get("role_applicability") or ""),
        "tags": str(row.get("tags") or ""),
        "quality": int(row.get("quality") or 3),
        "recommended_language": str(row.get("recommended_language") or ""),
        "notes": str(row.get("notes") or ""),
    }
    return clause, meta

async def ingest_csv(csv_path: str, batch: int = 128):
    df = pd.read_csv(csv_path)
    _ensure_cols(df)

    embedder = EmbedderService()
    qdrant = QdrantClient()

    clauses: List[ClauseNormalized] = []
    metas: List[dict] = []
    texts: List[str] = []

    for _, r in df.iterrows():
        clause, meta = _row_to_clause_and_meta(r)
        if clause is None:
            continue
        clauses.append(clause)
        metas.append(meta)
        texts.append(clause.analysis_text)

    total = len(clauses)
    if total == 0:
        print("No rows to ingest.")
        return

    print(f"Ingesting {total} reference snippets from {csv_path} ...")
    for i in range(0, total, batch):
        chunk_clauses = clauses[i:i+batch]
        chunk_texts = texts[i:i+batch]
        chunk_metas = metas[i:i+batch]

        embs = await embedder.create_embeddings(chunk_texts)
        for cl, emb, meta in zip(chunk_clauses, embs, chunk_metas):
            await qdrant.upsert_clauses(        # ← 여기!
                contract_id=meta["contract_id"],
                clauses=[cl],
                embeddings=[emb],
                meta=meta,
            )
        print(f"  - upserted {min(i+batch, total)}/{total}")

    print(f"✅ Done. Ingested {total} rows.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_refs_csv.py <csv_path>")
        sys.exit(1)
    asyncio.run(ingest_csv(sys.argv[1]))
