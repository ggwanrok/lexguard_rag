# app/services/summarizer.py (Gemini 전용)
from typing import List, Dict, Any
import asyncio, json
import google.generativeai as genai
from app.config import settings

class SummarizerService:
    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY 필요(요약에 사용)")
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        self.max_tokens = 900
        self.temperature = 0.1

    def _risk_stats(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(items)
        highs = sum(1 for x in items if str(x.get("risk_assessment", {}).get("risk_level", "")).upper() in {"HIGH","CRITICAL"})
        return {"total": total, "highs": highs}

    async def generate_summary(self, clause_items: List[Dict[str, Any]]) -> str:
        stats = self._risk_stats(clause_items)
        # clause_items에는 각 조항의 risk_level/score/why/reco 등이 들어옴
        prompt = f"""
당신은 계약서 리스크 컨설턴트입니다. 아래 조항 분석 결과를 바탕으로 한국어로 간결한 요약을 작성하세요.

- 전체 조항 수: {stats['total']}, High/ Critical 개수: {stats['highs']}
- 형식:
1) 전체 평가(2-3문장)
2) 주요 위험 요소(3-5개, 불릿)
3) 개선 권장(3-5개, 불릿)
4) 결론(1-2문장)

[조항 분석 결과(JSON)]
{json.dumps(clause_items, ensure_ascii=False)}
"""
        def _call():
            resp = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                ),
            )
            return (resp.text or "").strip()
        return await asyncio.to_thread(_call)
