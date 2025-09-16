import asyncio
import json
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from pydantic import BaseModel, Field

from app.config import settings
from app.models.schema import RiskAssessment, Citation
from app.services.prompt_builder import PromptBuilder

class _RiskSchema(BaseModel):
    risk_level: str = Field(description="LOW|MEDIUM|HIGH|CRITICAL")
    risk_score: float = Field(ge=0, le=100)
    risk_factors: List[str]
    recommendations: List[str]
    explanation: str

class LLMEvaluator:
    """Gemini 기반 평가기. 구조화(JSON) 출력과 이벤트 루프 비차단 보장."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY 필요")
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature

    def _citations_from_refs(self, refs: List[Dict[str, Any]]) -> List[Citation]:
        cits: List[Citation] = []
        for r in refs[:5]:
            md = r.get("metadata", {}) or {}
            cits.append(
                Citation(
                    chunk_id=r.get("chunk_id", ""),
                    source=md.get("issuer") or md.get("contract_type"),
                    clause_number=md.get("original_identifier"),
                    similarity=float(r.get("similarity_score", 0.0)),
                )
            )
        return cits

    async def _generate_json(self, prompt: str) -> dict:
        """Call Gemini in a thread and parse JSON-only responses."""
        def _call():
            return self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_mime_type="application/json",
                ),
            )

        resp = await asyncio.to_thread(_call)  # non-blocking
        text = (getattr(resp, "text", None) or "").strip()
        try:
            return json.loads(text)
        except Exception:
            # keep logs in caller; avoid leaking raw LLM text here
            raise ValueError("LLM JSON parse failed")

    async def evaluate_clause_risk_with_refs(
        self,
        clause_text: str,
        references: List[Dict[str, Any]],
        contract_type: Optional[str],
        jurisdiction: Optional[str],
    ) -> RiskAssessment:
        refs_text = PromptBuilder.format_references(references)
        prompt = PromptBuilder.clause_risk_prompt(
            clause_text, refs_text, contract_type or "", jurisdiction or ""
        )
        try:
            data = await self._generate_json(prompt)
            parsed = _RiskSchema(**data)
            ra = RiskAssessment(
                risk_level=parsed.risk_level.upper(),
                risk_score=float(parsed.risk_score),
                risk_factors=parsed.risk_factors,
                recommendations=parsed.recommendations,
                explanation=parsed.explanation,
                citations=self._citations_from_refs(references),
            )
            return ra
        except Exception:
            return RiskAssessment(
                risk_level="MEDIUM",
                risk_score=50.0,
                risk_factors=["LLM 응답 파싱 실패"],
                recommendations=["전문가 검토 권장"],
                explanation="구조화 응답 실패",
                citations=self._citations_from_refs(references),
            )

    async def evaluate_overall_risk(self, contract_text: str) -> RiskAssessment:
        prompt = PromptBuilder.overall_risk_prompt(contract_text)
        try:
            data = await self._generate_json(prompt)
            parsed = _RiskSchema(**data)
            return RiskAssessment(
                risk_level=parsed.risk_level.upper(),
                risk_score=float(parsed.risk_score),
                risk_factors=parsed.risk_factors,
                recommendations=parsed.recommendations,
                explanation=parsed.explanation,
                citations=[],
            )
        except Exception:
            return RiskAssessment(
                risk_level="MEDIUM",
                risk_score=50.0,
                risk_factors=["LLM 응답 파싱 실패"],
                recommendations=["전문가 검토 권장"],
                explanation="구조화 응답 실패",
                citations=[],
            )
