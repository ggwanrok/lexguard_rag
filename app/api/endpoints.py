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

def _is_sufficient_text(txt: Optional[str]) -> bool:
    if not txt:
        return False
    t = (txt or "").strip().lower()
    # 기본: 오케스트레이터의 판단기를 우선 사용
    if _is_sufficient(txt):
        return True
    # 방어적 추가 매칭(영/한 변형)
    suff_variants = {
        _SUFF.strip().lower(),
        "sufficient as-is.",
        "sufficient as-is",
        "sufficient as is",
        "as is sufficient",
        "현 상태로 충분합니다.",
        "현 상태로 충분합니다",
        "현 상태로 충분",
        "충분합니다.",
        "충분합니다",
    }
    return t in suff_variants

def _force_sufficient_only(ra: RiskAssessment) -> RiskAssessment:
    """LOW/UNKNOWN 레벨에서 '충분'류가 존재할 때, 모든 조언 제거 후 '충분'만 1개 남김."""
    return RiskAssessment(
        risk_level=ra.risk_level,
        risk_score=ra.risk_score,
        risk_factors=ra.risk_factors or [],
        recommendations=[_SUFF],  # 충분 메시지 한 줄만
        explanation=None if _is_sufficient_text(getattr(ra, "explanation", None)) else getattr(ra, "explanation", None),
        citations=ra.citations or [],
    )

def _strip_sufficient_everywhere(ra: RiskAssessment) -> RiskAssessment:
    """MEDIUM/HIGH/CRITICAL 등에서 충분류 문구를 모든 필드에서 제거."""
    # 1) 추천
    recos = [r for r in (ra.recommendations or []) if not _is_sufficient_text(r)]
    # 2) 설명
    expl = getattr(ra, "explanation", None)
    if _is_sufficient_text(expl):
        expl = None
    # 3) why(있을 경우)
    why = getattr(ra, "why", None)
    if isinstance(why, list):
        why = [w for w in why if not _is_sufficient_text(w)]
    # 4) risk_factors(혹시 충분류가 섞여 있으면 제거)
    rfs = [f for f in (ra.risk_factors or []) if not _is_sufficient_text(f)]

    new_ra = RiskAssessment(
        risk_level=ra.risk_level,
        risk_score=ra.risk_score,
        risk_factors=rfs,
        recommendations=recos,
        explanation=expl,
        citations=ra.citations or [],
    )
    # 선택 필드(why)가 모델에 존재한다면 다시 달아줌
    try:
        setattr(new_ra, "why", why)
    except Exception:
        pass
    return new_ra

def _harden_sufficient_policy(resp: ContractAnalysisResponse) -> ContractAnalysisResponse:
    """
    하드닝 정책(요청사항 정확 반영):
    - 충분류가 하나라도 존재할 때:
        * risk_level ∈ {LOW, UNKNOWN} → 기존 조언 모두 제거하고 '충분' 한 줄만 남김
        * 그 외 레벨(MEDIUM/HIGH/CRITICAL) → 충분류 문구를 추천/설명/why/리스크요인 등 전 필드에서 제거
    - 충분류가 없으면 변경 없음
    - revised_text가 비어있으면 '_SUFF'로 보강
    """
    hardened_items = []
    for ci in resp.clause_analysis:
        ra = ci.risk_assessment
        # 충분류 존재 여부는 추천/설명/why/리스크요인 전반에서 탐지
        has_sufficient = any([
            any(_is_sufficient_text(r) for r in (ra.recommendations or [])),
            _is_sufficient_text(getattr(ra, "explanation", None)),
            any(_is_sufficient_text(w) for w in (getattr(ra, "why", None) or [])) if isinstance(getattr(ra, "why", None), list) else False,
            any(_is_sufficient_text(f) for f in (ra.risk_factors or [])),
        ])

        if has_sufficient:
            level = (ra.risk_level or "").upper()
            if level in ("LOW", "UNKNOWN"):
                ra = _force_sufficient_only(ra)
            else:
                ra = _strip_sufficient_everywhere(ra)

        revised_text = ci.revised_text or _SUFF  # 카드 누락 방지

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

    # Overall도 동일 정책 적용
    oa = resp.overall_risk_assessment
    has_sufficient_overall = any([
        any(_is_sufficient_text(r) for r in (oa.recommendations or [])),
        _is_sufficient_text(getattr(oa, "explanation", None)),
        any(_is_sufficient_text(w) for w in (getattr(oa, "why", None) or [])) if isinstance(getattr(oa, "why", None), list) else False,
        any(_is_sufficient_text(f) for f in (oa.risk_factors or [])),
    ])
    if has_sufficient_overall:
        olevel = (oa.risk_level or "").upper()
        if olevel in ("LOW", "UNKNOWN"):
            resp.overall_risk_assessment = _force_sufficient_only(oa)
        else:
            resp.overall_risk_assessment = _strip_sufficient_everywhere(oa)
    else:
        resp.overall_risk_assessment = oa

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
        ident = getattr(c, "original_identifier", None) or c.clause_id
        lines.append(f"\n### {ident} — {r.risk_level} {r.risk_score}")
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
            + "\n".join(f"- {r}" for r in (overall.recommendations or []))
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
    resp = _harden_sufficient_policy(resp)

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
