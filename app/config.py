# app/config.py
from typing import Optional, Literal
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
    api_version: str = "1.6.3"

    # === Qdrant ===
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "contract_embeddings"

    # === Embedding provider switch ===
    # "huggingface" | "openai"
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"

    # --- Hugging Face (local) ---
    hf_model_name: str = "kakao1513/KURE-legal-ft-v1"
    hf_device: str = "cpu"   # "cuda", "mps", or "cpu"

    # --- OpenAI (API) ---
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"

    # 공통: 차원. 0 또는 음수면 모델에서 자동 탐지(가능한 경우)
    embedding_dim: int = 0

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
    risk_score_offset: float = 1.5
    risk_uplift_trigger: float = 3.5
    risk_uplift_citation: float = 2.5
    risk_uplift_why_per_2: float = 1.0
    risk_uplift_cap: float = 16.0
    risk_floor_enabled: bool = False

    # === Recommendation relevance ===
    reco_rel_min_overlap: int = 1
    reco_rel_min_len: int = 18

    # === Rewrite ===
    rewrite_enabled: bool = True
    rewrite_return_original_when_none: bool = False
    rewrite_min_ratio: float = 0.3
    rewrite_min_chars: int = 20

    # === Highlight spans ===
    highlight_max_spans: int = 8
    highlight_min_chars: int = 3

    # === UNKNOWN 관리 ===
    unknown_maps_to_zero: bool = True

    # === 점수 감쇠 ===
    score_damping_cat: float = 0.92
    score_damping_clause: float = 0.92
    score_damping_overall: float = 0.94

    # === 병합 가중치 ===
    merge_topn: int = 3
    merge_weight_max: float = 0.65
    merge_weight_topn: float = 0.35

    overall_topn: int = 4
    overall_weight_max: float = 0.65
    overall_weight_topn: float = 0.35

    # === Evidence 기반 감쇠 ===
    ev_mult_0: float = 0.60
    ev_mult_1: float = 0.80
    ev_mult_2plus: float = 1.00

    # === 미드밴드 눌림 ===
    mid_pull: float = 0.85

    # === 목적 조항 LOW 상한 캡 ===
    purpose_low_cap: float = 28.0
    purpose_ev_threshold: int = 1

settings = Settings()
