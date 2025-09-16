# app/config.py
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # === API ===
    api_title: str = "RAG Contract Audit API"
    api_version: str = "1.6.2"

    # === Qdrant ===
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "contract_embeddings"

    # === Embedding (OpenAI v1) ===
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # === LLM (Gemini) ===
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-pro"

    # === Chunking / Retrieval ===
    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_top_k: int = 6
    min_similarity: float = 0.78

    # === LLM generation ===
    max_tokens: int = 2000
    temperature: float = 0.1
    gemini_concurrency: int = 4
    gemini_timeout_sec: int = 25

    # === Risk score calibration (분포↓ + 편차↑) ===
    risk_score_offset: float = 1.5          # 기본 상향을 낮춰 전체 분포 하향
    risk_uplift_trigger: float = 3.5        # 증거 있을 때만 상향
    risk_uplift_citation: float = 2.5
    risk_uplift_why_per_2: float = 1.0
    risk_uplift_cap: float = 16.0           # 보정 상한

    risk_floor_enabled: bool = False        # floor 해제 → 낮은 값은 더 낮게

    # === Recommendation relevance ===
    reco_rel_min_overlap: int = 1
    reco_rel_min_len: int = 18

    # === Clause rewriter ===
    rewrite_enabled: bool = True
    rewrite_return_original_when_none: bool = False
    rewrite_min_ratio: float = 0.3
    rewrite_min_chars: int = 20

    # === Highlight spans ===
    highlight_max_spans: int = 8
    highlight_min_chars: int = 3

    # === UNKNOWN 관리 ===
    unknown_maps_to_zero: bool = True

    # === 점수 감쇠(전체 살짝 낮춤) ===
    score_damping_cat: float = 0.92
    score_damping_clause: float = 0.92
    score_damping_overall: float = 0.94

    # === 병합 가중치(피크는 살리고 평균 영향 완만) ===
    merge_topn: int = 3
    merge_weight_max: float = 0.65
    merge_weight_topn: float = 0.35

    overall_topn: int = 4
    overall_weight_max: float = 0.65
    overall_weight_topn: float = 0.35

    # === Evidence 기반 감쇠(증거 없으면 더 낮추기) ===
    ev_mult_0: float = 0.60   # triggers=0, citations=0, why<2
    ev_mult_1: float = 0.80   # 위 1개만 충족
    ev_mult_2plus: float = 1.00  # 2개 이상 충족

    # === 미드밴드(40~60) 눌림 계수 ===
    mid_pull: float = 0.85

    # === 목적 조항 LOW 상한 캡(증거 빈약 시) ===
    purpose_low_cap: float = 28.0           # 목적 조항 최대점수
    purpose_ev_threshold: int = 1           # ev(트리거/인용/why2+) <= 1이면 상한 적용

settings = Settings()
