from typing import List, Dict, Any

class PromptBuilder:
    @staticmethod
    def format_references(refs: List[Dict[str, Any]]) -> str:
        if not refs:
            return "관련 참고 자료 없음."
        lines = []
        for i, r in enumerate(refs, 1):
            md = r.get("metadata", {})
            src = md.get("issuer") or md.get("contract_type") or "reference"
            cnum = md.get("original_identifier") or md.get("clause_id")
            content = (r.get("content", "") or "").replace("\n", " ")[:400]
            lines.append(f"[{i}] 출처:{src} / 조항:{cnum}\n{content}")
        return "\n\n".join(lines)

    @staticmethod
    def clause_risk_prompt(
        clause_text: str,
        references_text: str,
        contract_type: str = "",
        jurisdiction: str = "",
    ) -> str:
        # Gemini에서 JSON 강제 출력되므로 본문은 맥락 위주로 전달
        return f"""
당신은 계약서 위험도 평가 전문가입니다.
아래 '검토 대상 조항'을 '참고 자료'에 근거하여 평가하세요.
가능한 한 참고 자료의 표현을 재인용하고, 없을 경우 '근거 부족'으로 명시하세요.

[계약 맥락]
- contract_type: {contract_type}
- jurisdiction: {jurisdiction}

[검토 대상 조항]
{clause_text}

[참고 자료]
{references_text}

반환은 JSON 한 개 객체로만 수행합니다.
{{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "risk_score": 0-100,
  "risk_factors": ["..."],
  "recommendations": ["..."],
  "explanation": "요약 근거 포함"
}}
        """

    @staticmethod
    def overall_risk_prompt(contract_text: str) -> str:
        return f"""
다음 계약서 전체를 종합적으로 평가하고 JSON만 반환하세요.
계약서 내용(일부):
{contract_text[:3800]}

스키마:
{{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "risk_score": 0-100,
  "risk_factors": ["..."],
  "recommendations": ["..."],
  "explanation": "전체 근거 요약"
}}
        """
