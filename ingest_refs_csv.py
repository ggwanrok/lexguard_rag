#!/usr/bin/env python3
import asyncio
import uuid
import sys
from typing import List, Optional
import pandas as pd
import math

from app.services.embedder import EmbedderService
from app.services.qdrant_client import QdrantClient
from app.models.schema import ClauseNormalized

REQUIRED_COLS = ["text"]

def _ensure_cols(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

def _nz(s: Optional[str]) -> str:
    return ("" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)).strip()

def _row_to_clause_and_meta(row: pd.Series):
    text = _nz(row.get("text"))
    if not text:
        return None, None

    identifier = _nz(row.get("identifier") or row.get("title") or row.get("category") or "ref")
    title = _nz(row.get("title"))
    clause_type = _nz(row.get("clause_type") or "article")

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

    lang = _nz(row.get("recommended_language") or "ko")
    meta = {
        "contract_id": f"ref_{uuid.uuid4().hex[:12]}",
        "contract_name": _nz(row.get("source") or "Reference"),
        "contract_type": _nz(row.get("contract_type") or "NDA"),
        "issuer": _nz(row.get("source") or "ReferenceLib"),
        "jurisdiction": _nz(row.get("jurisdiction") or "KR"),
        "language": lang,
        "subject": "NDA",
        "version": _nz(row.get("version")),
        "source_uri": None,
        "is_reference": bool(row.get("is_reference", True)),
        "category": _nz(row.get("category")),
        "subcategory": _nz(row.get("subcategory")),
        "role_applicability": _nz(row.get("role_applicability")),
        "tags": _nz(row.get("tags")),
        "quality": int(row.get("quality") or 3),
        "recommended_language": lang,
        "notes": _nz(row.get("notes")),
    }
    return clause, meta

async def ingest_csv(csv_path: str, batch: int = 128, use_batch_upsert: bool = False, reset: bool = False):
    df = pd.read_csv(csv_path)
    _ensure_cols(df)

    embedder = EmbedderService()
    qdrant = QdrantClient()
    
    # --reset 옵션: 기존 컬렉션 삭제 후 재생성
    if reset:
        print(f"🗑️  Resetting collection '{qdrant.collection}'...")
        await qdrant.reset_collection()
        print("✅ Collection reset complete.")

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

    # 1) 임베딩 먼저 전부(또는 배치로) 구함
    all_embs: List[List[float]] = []
    for i in range(0, total, batch):
        chunk_texts = texts[i:i+batch]
        embs = await embedder.create_embeddings(chunk_texts)
        all_embs.extend(embs)
        print(f"  - embedded {min(i+batch, total)}/{total}")

    # 2) Qdrant 업서트
    for i in range(0, total, batch):
        chunk = list(zip(clauses[i:i+batch], all_embs[i:i+batch], metas[i:i+batch]))
        if use_batch_upsert and hasattr(qdrant, "upsert_many"):
            await qdrant.upsert_many(chunk)
        else:
            for cl, emb, meta in chunk:
                await qdrant.upsert_clauses(
                    contract_id=meta["contract_id"],
                    clauses=[cl],
                    embeddings=[emb],
                    meta=meta,
                )
        print(f"  - upserted {min(i+batch, total)}/{total}")

    print(f"✅ Done. Ingested {total} rows.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_refs_csv.py <csv_path> [--batch-upsert] [--reset]")
        print("  --reset: 기존 컬렉션을 삭제하고 새로 시작합니다")
        sys.exit(1)
    csv = sys.argv[1]
    use_batch = ("--batch-upsert" in sys.argv[2:])
    reset = ("--reset" in sys.argv[2:])
    asyncio.run(ingest_csv(csv, use_batch_upsert=use_batch, reset=reset))
