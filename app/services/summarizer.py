from typing import List, Dict, Any
from openai import AsyncOpenAI
from app.config import settings

class SummarizerService:
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY 필요(요약에 사용)")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o-mini"
        self.max_tokens = 900
        self.temperature = 0.1

    def _risk_stats(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(items)
        dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        score_sum = 0.0
        for it in items:
            ra = it["risk_assessment"]
            level = (getattr(ra, "risk_level", None) or "MEDIUM").upper()
            if level not in dist:
                level = "MEDIUM"
            dist[level] += 1
            score_sum += float(getattr(ra, "risk_score", 0.0) or 0.0)
        avg = round(score_sum / total, 2) if total else 0.0
        high_pct = round(((dist["HIGH"] + dist["CRITICAL"]) / total) * 100, 2) if total else 0.0
        return {"total": total, "dist": dist, "avg": avg, "high_pct": high_pct}

    async def generate_summary(self, contract_text: str, clause_items: List[Dict[str, Any]]) -> str:
        stats = self._risk_stats(clause_items)
        prompt = f"""
다음 분석 결과를 경영진 보고용으로 간결히 요약하세요.
- 총 조항 수: {stats['total']}
- 평균 위험 점수: {stats['avg']}
- HIGH/CRITICAL 비율: {stats['high_pct']}%
- 분포: {stats['dist']}

원칙:
1) 모든 문장은 한국어 격식체(합니다체)로 작성하십시오.
2) "~요/~세요/~해요/명사형 어미(…함/…임)"를 사용하지 마십시오.
3) 권고 문장은 "…을 권장합니다/…해야 합니다/…바람직합니다."로 끝맺으십시오.

형식:
1) 전체 평가(2-3문장)
2) 주요 위험 요소(3-5개, 불릿)
3) 개선 권장(3-5개, 불릿)
4) 결론(1-2문장)
"""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()
