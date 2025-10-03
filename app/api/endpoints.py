# app/api/endpoints.py
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Query
from fastapi.responses import PlainTextResponse
from datetime import datetime
from typing import Optional
import uuid

from app.models.schema import (
    ContractUploadRequest,
    ContractAnalysisResponse,
    ClauseAnalysisItem,
    SearchQuery,
    SearchResponse,
    ContractNormalized,
    RiskAssessment,
)
from app.services.normalizer import NormalizerService
from app.services.retriever import RetrieverService
from app.services.summarizer import SummarizerService
from app.services.audit_orchestrator import AuditOrchestrator, _SUFF, _is_sufficient
from app.utils.logger import logger

router = APIRouter()

# Service singletons
normalizer = NormalizerService()
retriever = RetrieverService()
summarizer = SummarizerService()
auditor = AuditOrchestrator()


# --------------------------- 공통 유틸 ---------------------------

def _force_unknown(ra: RiskAssessment) -> RiskAssessment:
    """UNKNOWN/0점으로 강제(충분 메시지 포함)."""
    return RiskAssessment(
        risk_level="UNKNOWN",
        risk_score=0.0,
        risk_factors=[],
        recommendations=[_SUFF],
        explanation="Sufficient as-is.",
        citations=[],
    )

def _harden_unknown_invariants(resp: ContractAnalysisResponse) -> ContractAnalysisResponse:
    """
    백엔드 최종 응답 직전 하드닝:
    - recommendations가 전부 '충분'류이거나, risk_level이 UNKNOWN이거나, risk_score<=0 이면 → UNKNOWN/0으로 강제 동기화
    - revised_text가 비어있으면(UNKNOWN 포함) '_SUFF'를 채워 개정안 카드 누락 방지
    - overall도 동일 규칙 적용(추천이 모두 '충분'이면 [_SUFF]만 남기고 UNKNOWN/0)
    """
    hardened_items = []
    for ci in resp.clause_analysis:
        ra = ci.risk_assessment
        recos = ra.recommendations or []
        all_sufficient = (len(recos) == 0) or all(_is_sufficient(r) for r in recos)
        if ra.risk_level == "UNKNOWN" or ra.risk_score <= 0.0 or all_sufficient:
            ra = _force_unknown(ra)
        # 개정문 누락 방지
        revised_text = ci.revised_text or _SUFF
        hardened_items.append(
            ClauseAnalysisItem(
                clause_id=ci.clause_id,
                original_identifier=ci.original_identifier,
                original_text=ci.original_text,
                risk_assessment=ra,
                revised_text=revised_text,
                revised_spans=ci.revised_spans,
            )
        )

    resp.clause_analysis = hardened_items

    # overall 하드닝
    oa = resp.overall_risk_assessment
    orec = oa.recommendations or []
    o_all_sufficient = (len(orec) == 0) or all(_is_sufficient(r) for r in orec)
    if oa.risk_level == "UNKNOWN" or oa.risk_score <= 0.0 or o_all_sufficient:
        resp.overall_risk_assessment = _force_unknown(oa)

    return resp


def _build_markdown_report(res: ContractAnalysisResponse) -> str:
    """응답을 마크다운 보고서 텍스트로 변환."""
    oa = res.overall_risk_assessment
    lines = []
    title = res.contract_name or res.normalized.contract_name or "Contract"
    lines.append(f"# Risk Report — {title}")
    lines.append(f"- **Level:** {oa.risk_level}  |  **Score:** {oa.risk_score}")
    if oa.risk_factors:
        lines.append("\n## Key Risks")
        lines += [f"- {x}" for x in oa.risk_factors]
    if oa.recommendations:
        lines.append("\n## Recommendations")
        lines += [f"- {x}" for x in oa.recommendations]
    lines.append("\n## Clause Details")
    for c in res.clause_analysis:
        r = c.risk_assessment
        lines.append(f"\n### {c.get('original_identifier', None) or c.clause_id} — {r.risk_level} {r.risk_score}")
        why = getattr(r, "why", None)
        if why:
            lines.append("**Why**")
            lines += [f"- {w}" for w in why[:4]]
        if r.recommendations:
            lines.append("**Fix**")
            lines += [f"- {x}" for x in r.recommendations[:4]]
    lines.append("\n---\n*Generated at " + res.analysis_timestamp + "*")
    return "\n".join(lines)


async def _analyze_core(request: ContractUploadRequest) -> ContractAnalysisResponse:
    """Normalize → Per-clause × category audits → Aggregate → Summarize → Response"""
    # 1) Normalize
    contract_id = str(uuid.uuid4())
    cn: ContractNormalized = normalizer.normalize(
        raw_text=request.raw_text,
        contract_id=contract_id,
        meta={
            "contract_name": request.contract_name,
            "contract_type": request.contract_type,
            "issuer": request.issuer,
            "jurisdiction": request.jurisdiction,
            "language": request.language,
            "subject": request.subject,
            "version": request.version,
            "source_uri": request.source_uri,
            "is_reference": False,
        },
    )

    # 2) Analyze
    ctype = request.contract_type or "NDA"
    juri = request.jurisdiction or "KR"
    role = "recipient"
    overall, clause_items = await auditor.analyze(
        cn, contract_type=ctype, jurisdiction=juri, role=role
    )

    # 3) Summarize
    try:
        summary = await summarizer.generate_summary([ci.model_dump() for ci in clause_items])
    except Exception:
        summary = (
            f"Overall: {overall.risk_level} ({overall.risk_score})\n"
            + "Key risks: "
            + (", ".join(overall.risk_factors) if overall.risk_factors else "-")
            + "\n"
            + "\n".join(f"- {r}" for r in overall.recommendations)
        )

    # 4) Build response + 최종 하드닝
    resp = ContractAnalysisResponse(
        contract_id=cn.contract_id,
        contract_name=request.contract_name,
        analysis_timestamp=datetime.utcnow().isoformat() + "Z",
        overall_risk_assessment=overall,
        clause_analysis=clause_items,
        summary=summary,
        normalized=cn,
    )
    resp = _harden_unknown_invariants(resp)

    logger.info(
        f"analyze ok | contract_id={cn.contract_id} clauses={len(cn.clauses)}"
    )
    return resp


# --------------------------- 기존 JSON 라우트 ---------------------------

@router.post("/contracts/analyze", response_model=ContractAnalysisResponse)
async def analyze_contract(request: ContractUploadRequest):
    """
    JSON 입력(raw_text) → JSON 응답(ContractAnalysisResponse).
    """
    try:
        return await _analyze_core(request)
    except Exception:
        err_id = str(uuid.uuid4())
        logger.exception(f"analyze failed | error_id={err_id}")
        raise HTTPException(
            status_code=500,
            detail=f"분석 처리 중 오류가 발생했습니다. 오류코드: {err_id}",
        )


# --------------------------- text/plain 라우트 ---------------------------

@router.post("/contracts/analyze-text")
async def analyze_text(
    raw_body: bytes = Body(..., media_type="text/plain"),
    contract_type: str = Query("NDA"),
    jurisdiction: str = Query("KR"),
    language: str = Query("ko"),
    contract_name: Optional[str] = Query(None),
    format: str = Query("json", description="응답 형식: json|markdown|plain"),
):
    """
    텍스트 그대로(text/plain) 받아 분석.
    format:
      - json     → JSON 응답 (ContractAnalysisResponse)
      - markdown → 마크다운 보고서(text/markdown)
      - plain    → 일반 텍스트(text/plain)
    """
    try:
        req = ContractUploadRequest(
            raw_text=raw_body.decode("utf-8", "ignore"),
            contract_type=contract_type,
            jurisdiction=jurisdiction,
            language=language,
            contract_name=contract_name,
        )
        res = await _analyze_core(req)
        if format.lower() == "json":
            return res
        md = _build_markdown_report(res)
        if format.lower() == "markdown":
            return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")
        # plain
        return PlainTextResponse(md, media_type="text/plain; charset=utf-8")
    except Exception:
        err_id = str(uuid.uuid4())
        logger.exception(f"analyze-text failed | error_id={err_id}")
        raise HTTPException(
            status_code=500,
            detail=f"분석 처리 중 오류가 발생했습니다. 오류코드: {err_id}",
        )


# --------------------------- 파일 업로드 라우트 ---------------------------

@router.post("/contracts/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    contract_type: str = Query("NDA"),
    jurisdiction: str = Query("KR"),
    language: str = Query("ko"),
    contract_name: Optional[str] = Query(None),
    format: str = Query("json", description="응답 형식: json|markdown|plain"),
):
    """
    multipart/form-data 파일 업로드로 분석.
    (TXT/MD 등 텍스트 파일 권장)
    """
    try:
        raw = (await file.read()).decode("utf-8", "ignore")
        req = ContractUploadRequest(
            raw_text=raw,
            contract_type=contract_type,
            jurisdiction=jurisdiction,
            language=language,
            contract_name=contract_name or file.filename,
        )
        res = await _analyze_core(req)
        if format.lower() == "json":
            return res
        md = _build_markdown_report(res)
        if format.lower() == "markdown":
            return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")
        return PlainTextResponse(md, media_type="text/plain; charset=utf-8")
    except Exception:
        err_id = str(uuid.uuid4())
        logger.exception(f"analyze-file failed | error_id={err_id}")
        raise HTTPException(
            status_code=500,
            detail=f"파일 분석 중 오류가 발생했습니다. 오류코드: {err_id}",
        )


# --------------------------- 검색 ---------------------------

@router.post("/search", response_model=SearchResponse)
async def search_contracts(query: SearchQuery):
    try:
        refs = await retriever.retrieve_similar_chunks(
            query.query,
            top_k=query.top_k,
            reference_only=False,
        )
        return SearchResponse(query=query.query, results=refs, total_results=len(refs))
    except Exception:
        err_id = str(uuid.uuid4())
        logger.exception(f"search failed | error_id={err_id}")
        raise HTTPException(
            status_code=500,
            detail=f"검색 처리 중 오류가 발생했습니다. 오류코드: {err_id}",
        )
