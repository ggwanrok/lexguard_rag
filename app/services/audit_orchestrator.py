# app/services/audit_orchestrator.py
import asyncio
import json
import re
import time
from typing import List, Dict, Tuple, Any
from jinja2 import Template
import httpx
import google.generativeai as genai

from app.config import settings
from app.models.schema import ContractNormalized, ClauseAnalysisItem, RiskAssessment, DiffSpan
from app.packs.loader import load_pack
from app.services.retriever import RetrieverService
from app.services.prompt_builder import PromptBuilder
from app.utils.logger import logger
from app.utils.diff import compute_revised_spans

_LEVEL = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# ==================== helpers ====================

GENERIC_MARKERS = ("일반적으로", "바람직", "권장", "필요", "적절", "고려")
DOMAIN_KEYWORDS = ("이메일", "통지", "유예", "사전 동의", "제3자", "양도", "비밀", "반환", "파기", "목적 외")

def _looks_generic(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 12:
        return True
    return sum(1 for k in GENERIC_MARKERS if k in t) >= 2

def _level_from_score(score: float) -> str:
    s = max(0.0, min(100.0, float(score)))
    if s >= 75: return "CRITICAL"
    if s >= 60: return "HIGH"
    if s >= 40: return "MEDIUM"
    return "LOW"

# --- Header lock: '제n조(표제)' 보존 ---
_HEADER_RE = re.compile(r'^\s*(제\s*\d+\s*조\s*(?:\([^)]+\))?)')
def _extract_header(text: str) -> tuple[str, int]:
    m = _HEADER_RE.match(text or "")
    if not m:
        return ("", 0)
    return (m.group(1).strip(), m.end())

def _enforce_header(original: str, revised: str) -> str:
    o_hdr, _ = _extract_header(original)
    if not o_hdr:
        return revised
    r_hdr, r_end = _extract_header(revised)
    if r_hdr:
        if r_hdr != o_hdr:
            return f"{o_hdr}{revised[r_end:]}"
        return revised
    return f"{o_hdr} {revised.lstrip()}"

def _filter_spans_after_header(revised: str, spans: List[Dict]) -> List[Dict]:
    _, hdr_end = _extract_header(revised)
    if hdr_end <= 0 or not spans:
        return spans or []
    out = []
    for sp in spans:
        start, end = sp["start"], sp["end"]
        if end <= hdr_end:
            continue
        if start < hdr_end:
            start = hdr_end
        if start < end:
            out.append({"kind": sp["kind"], "start": start, "end": end, "text": revised[start:end]})
    return out

def _select_major_spans(revised: str, spans: List[Dict]) -> List[Dict]:
    """가독성을 위해 길이 기준으로 상위 n개만 선택(겹침 제거)."""
    if not spans:
        return []
    max_n = int(getattr(settings, "highlight_max_spans", 8))
    min_chars = int(getattr(settings, "highlight_min_chars", 3))
    cand = [s for s in spans if (s["end"] - s["start"]) >= min_chars]
    cand.sort(key=lambda s: (s["end"] - s["start"]), reverse=True)
    chosen: List[Dict] = []
    for sp in cand:
        if len(chosen) >= max_n:
            break
        overlap = any(not (sp["end"] <= c["start"] or sp["start"] >= c["end"]) for c in chosen)
        if not overlap:
            chosen.append(sp)
    chosen.sort(key=lambda s: s["start"])
    return chosen

# ---- Recommendation relevance (원문/리스크 맥락과 연결 검사) ----
_WORD_RE = re.compile(r"[A-Za-z0-9가-힣_/.-]+")
def _tokens(text: str) -> set[str]:
    return set(w for w in _WORD_RE.findall((text or "")) if len(w) >= 2)

def _reco_is_relevant(reco: str, clause_text: str, why: List[str], triggers: List[Dict]) -> bool:
    if not isinstance(reco, str):
        return False
    rtk = _tokens(reco)
    if not rtk:
        return False
    kw = set(DOMAIN_KEYWORDS)
    for w in (why or []): kw |= _tokens(w)
    for t in (triggers or []):
        if isinstance(t, dict) and t.get("text"): kw |= _tokens(t["text"])
    overlap = len(rtk & kw)
    min_overlap = int(getattr(settings, "reco_rel_min_overlap", 1))
    min_len = int(getattr(settings, "reco_rel_min_len", 18))
    if len(reco.strip()) < min_len:
        return False
    return overlap >= min_overlap

# ---- '지금으로도 충분합니다.' 판정(동의어/변형 포함) ----
_SUFF = "지금으로도 충분합니다."
_SUFF_PATTERNS = [
    r"\A[\"'“”‘’]?\s*지금으로도\s*충분합니다[\.!」']?\s*\Z",
    r"\A[\"'“”‘’]?\s*현재로도\s*충분합니다[\.!」']?\s*\Z",
    r"\A[\"'“”‘’]?\s*변경\s*필요\s*없습니다[\.!」']?\s*\Z",
    r"\A[\"'“”‘’]?\s*수정\s*(불요|불필요)입니다[\.!」']?\s*\Z",
    r"\A[\"'“”‘’]?\s*문제\s*없습니다[\.!」']?\s*\Z",
]
def _is_sufficient(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    for pat in _SUFF_PATTERNS:
        if re.match(pat, t):
            return True
    return False

def _decide_unknown(results: List[Dict]) -> bool:
    """모든 카테고리의 추천이 '충분'류(동의어/변형 포함)일 때 UNKNOWN."""
    has_any = False
    for r in results or []:
        has_any = True
        recos = [x for x in (r.get("recommendations") or []) if isinstance(x, str)]
        if any(not _is_sufficient(x) for x in recos):
            return False
    return has_any

# ---- 목적 조항 식별 + 증거강도 산출 ----
def _is_purpose_clause_text(text: str, original_identifier: str = "") -> bool:
    """조문 헤더/초반부에 '목적'이 있으면 목적 조항으로 간주."""
    try:
        hdr, _ = _extract_header(text)
    except Exception:
        hdr = ""
    head = (text or "")[:80]
    return ("목적" in hdr) or ("목적" in head)

def _evidence_strength(results: List[Dict]) -> int:
    """ev: triggers + citations + (why>=2). 카테고리별 누적."""
    ev = 0
    for r in results or []:
        if r.get("triggers"): ev += 1
        if r.get("citations"): ev += 1
        if len(r.get("why") or []) >= 2: ev += 1
    return ev

# ---- Risk score calibration (분포↓ + 편차↑) ----
def _calibrate_score(score: float, *, has_trigger: bool, has_citation: bool, why_count: int,
                     reco_count: int) -> float:
    base = float(score)
    base += float(getattr(settings, "risk_score_offset", 1.5))
    if has_trigger:  base += float(getattr(settings, "risk_uplift_trigger", 3.5))
    if has_citation: base += float(getattr(settings, "risk_uplift_citation", 2.5))
    base += (why_count // 2) * float(getattr(settings, "risk_uplift_why_per_2", 1.0))
    cap  = float(getattr(settings, "risk_uplift_cap", 16.0))
    uplifted = score + min(cap, max(0.0, base - score))

    # Floor(optional)
    if getattr(settings, "risk_floor_enabled", False):
        f2 = float(getattr(settings, "risk_floor_reco2", 45.0))
        f3 = float(getattr(settings, "risk_floor_reco3", 60.0))
        uplifted = max(f3 if reco_count>=3 else (f2 if reco_count>=2 else 0.0), uplifted)

    # Evidence multiplier
    ev = int(has_trigger) + int(has_citation) + (1 if why_count >= 2 else 0)
    if ev <= 0:   uplifted *= float(getattr(settings, "ev_mult_0", 0.60))
    elif ev == 1: uplifted *= float(getattr(settings, "ev_mult_1", 0.80))
    else:         uplifted *= float(getattr(settings, "ev_mult_2plus", 1.00))

    # Mid-band pull
    if 40.0 <= uplifted <= 60.0:
        uplifted *= float(getattr(settings, "mid_pull", 0.85))

    return max(0.0, min(100.0, uplifted))

# ==================== class ====================

class AuditOrchestrator:
    """Per-clause × per-category audits using Gemini with packs (taxonomy/checklists)."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY가 설정되어야 합니다.")
        genai.configure(api_key=settings.gemini_api_key)
        self.model_name = getattr(settings, "gemini_model", "gemini-1.5-pro")
        self.model = genai.GenerativeModel(self.model_name)
        self.api_key = settings.gemini_api_key
        self.temperature = getattr(settings, "temperature", 0.1)
        self.max_tokens = getattr(settings, "max_tokens", 2000)
        self.sem = asyncio.Semaphore(getattr(settings, "gemini_concurrency", 4))
        self.request_timeout = getattr(settings, "gemini_timeout_sec", 25)
        self.retriever = RetrieverService()

    # ---------- prompt ----------
    def _render_prompt(self, tpl_text: str, **kw) -> str:
        return Template(tpl_text, trim_blocks=True, lstrip_blocks=True).render(**kw)

    # ---------- LLM calls ----------
    def _genai_config(self):
        base = {"temperature": self.temperature, "max_output_tokens": self.max_tokens}
        try:
            return genai.types.GenerationConfig(**{**base, "response_mime_type": "application/json"})
        except TypeError:
            return genai.types.GenerationConfig(**base)

    def _rest_payload(self, prompt: str) -> Dict:
        return {"contents":[{"parts":[{"text":prompt}]}],
                "generationConfig":{"temperature":self.temperature,"maxOutputTokens":self.max_tokens}}

    async def _genai_generate(self, prompt: str) -> str:
        def _call():
            resp = self.model.generate_content(prompt, generation_config=self._genai_config())
            return (resp.text or "").strip()
        return await asyncio.to_thread(_call)

    async def _rest_generate(self, prompt: str) -> str:
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        headers={"x-goog-api-key":self.api_key,"Content-Type":"application/json"}
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            r=await client.post(url,headers=headers,json=self._rest_payload(prompt)); r.raise_for_status()
            data=r.json(); return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _generate(self, prompt: str) -> str:
        try: return await self._genai_generate(prompt)
        except Exception as e:
            logger.warning(f"[audit] gRPC failed; falling back to REST: {e}")
            return await self._rest_generate(prompt)

    # ---------- Category audit ----------
    async def _category_audit(
        self, tpl_text: str, contract_type: str, jurisdiction: str, role: str,
        category: str, clause_text: str, references_md: str
    ) -> Dict:
        prompt = self._render_prompt(
            tpl_text,
            contract_type=contract_type, jurisdiction=jurisdiction, role=role, category=category,
            checklist=load_pack(contract_type).checklist(category),
            clause_text=clause_text, references_md=references_md,
        )
        async with self.sem:
            try:
                logger.debug(f"[audit] start cat={category} len={len(clause_text)}")
                raw = await asyncio.wait_for(self._generate(prompt), timeout=self.request_timeout)
                logger.debug(f"[audit] done  cat={category} resp_len={len(raw)}")
            except asyncio.TimeoutError:
                logger.warning(f"[audit] timeout cat={category}"); return {}
            except Exception as e:
                logger.exception(f"[audit] generation error cat={category}: {e}"); return {}

        s=(raw or "").strip()
        if s.startswith("```"):
            s=re.sub(r"^```[\w-]*\n","",s); s=re.sub(r"\n```$","",s)
        data=None
        try: data=json.loads(s)
        except Exception:
            m=re.search(r"\{[\s\S]*\}$",s)
            if m:
                try: data=json.loads(m.group(0))
                except Exception: data=None
        if not isinstance(data,dict): return {}
        if data.get("irrelevant") is True: return {}

        risks=data.get("risks")
        item=risks[0] if isinstance(risks,list) and risks else data

        def _flo(x,df=0.0):
            try: return float(x)
            except Exception: return df

        out={
            "risk_level": str(item.get("risk_level","LOW")).upper(),
            "risk_score": _flo(item.get("risk_score", data.get("risk_score",0))),
            "risk_factors": list(item.get("risk_factors", data.get("risk_factors") or [])),
            "recommendations": list(item.get("recommendations", data.get("recommendations") or [])),
            "explanation": str(item.get("explanation") or ""),
        }

        # why → explanation 보완
        why=item.get("why") or []
        if why and not out["explanation"]:
            out["explanation"]="; ".join([str(w) for w in why])[:500]

        # triggers
        trig=[]
        for t in item.get("triggers") or []:
            if isinstance(t,dict) and {"start","end","text"}<=set(t.keys()):
                trig.append({"start":int(t["start"]),"end":int(t["end"]),"text":str(t["text"])})
        if trig: out["triggers"]=trig

        # citations
        cits=[]
        for c in item.get("citations") or []:
            if isinstance(c,dict) and "source" in c:
                cits.append({"source":str(c.get("source")),"identifier":str(c.get("identifier","")),
                             "note":str(c.get("note",""))})
        if cits: out["citations"]=cits

        # 추천안 품질/연관성 필터
        recos_raw=[r for r in (out.get("recommendations") or []) if isinstance(r,str)]
        recos_raw=[r.strip() for r in recos_raw if r.strip() and not _looks_generic(r)]
        relevant=[r for r in recos_raw if _reco_is_relevant(r, clause_text, why, trig)]
        if relevant:
            relevant=[r for r in relevant if not _is_sufficient(r)]
            out["recommendations"]=relevant[:5] if relevant else [_SUFF]
        else:
            out["recommendations"]=[_SUFF]

        # 카테고리 보정(감쇠는 config.score_damping_cat로 이미 포함됨)
        score=max(0.0, min(100.0, float(out.get("risk_score",0))))
        has_trigger=any((isinstance(t,dict) and t.get("text") and t["text"] in clause_text) for t in trig)
        has_citation=bool(cits)
        reco_count=len([r for r in out["recommendations"] if not _is_sufficient(r)])
        score=_calibrate_score(score, has_trigger=has_trigger, has_citation=has_citation,
                               why_count=len(why), reco_count=reco_count)
        out["risk_score"]=score
        out["risk_level"]=_level_from_score(score)
        return out

    # -------------------- Merge & Aggregate --------------------
    def _merge_clause(self, results: List[Dict], is_purpose: bool = False) -> RiskAssessment:
        if not results:
            return RiskAssessment(
                risk_level="LOW",
                risk_score=10.0,
                risk_factors=[],
                recommendations=[],
                explanation="No material risks detected.",
                citations=[],
            )

        # ZERO-LOCK: 모든 카테고리 추천이 '충분'류 → UNKNOWN/0 고정
        if _decide_unknown(results) and getattr(settings,"unknown_maps_to_zero",True):
            recs: List[str]=[]
            expls: List[str]=[]
            for r in results:
                for rc in (r.get("recommendations") or []):
                    if isinstance(rc,str): recs.append(rc.strip())
                if r.get("explanation"): expls.append(str(r["explanation"]))
            recs=list(dict.fromkeys(recs))[:5]
            explanation=("; ".join(recs) if recs else "; ".join(expls))[:500] or "Sufficient as-is."
            return RiskAssessment("UNKNOWN", 0.0, [], recs or [_SUFF], explanation, [])

        # 점수 집계(피크/평균 가중)
        scores=sorted([float(r.get("risk_score",0)) for r in results], reverse=True)
        topn_k=int(getattr(settings,"merge_topn",3))
        w_max=float(getattr(settings,"merge_weight_max",0.65))
        w_top=float(getattr(settings,"merge_weight_topn",0.35))
        topn=scores[:max(1,topn_k)]
        mean_topn=sum(topn)/max(1,len(topn))
        score=w_max*scores[0]+w_top*mean_topn
        level=_level_from_score(score)

        # 목적 조항 LOW 상한 캡(증거 약할 때만)
        if is_purpose:
            ev=_evidence_strength(results)
            ev_threshold=int(getattr(settings,"purpose_ev_threshold",1))
            if ev <= ev_threshold:
                cap=float(getattr(settings,"purpose_low_cap",28.0))
                if score > cap:
                    score=cap
                    level=_level_from_score(score)

        # 조항 감쇠
        score=max(0.0, min(100.0, score * float(getattr(settings,"score_damping_clause",0.92))))

        # 부가정보 병합
        factors: List[str]=[]
        recs: List[str]=[]
        expls: List[str]=[]
        cits: List[Any]=[]
        for r in results:
            for f in r.get("risk_factors",[]):
                if f and f not in factors: factors.append(f)
            for rc in r.get("recommendations",[]):
                if rc and rc not in recs: recs.append(rc)
            if r.get("explanation"): expls.append(str(r["explanation"]))
            for c in (r.get("citations") or []):
                if isinstance(c,dict): cits.append(c)

        return RiskAssessment(
            risk_level=level,
            risk_score=round(score,1),
            risk_factors=factors[:5],
            recommendations=recs[:5],
            explanation="; ".join(expls)[:500] if expls else "Aggregated from category audits.",
            citations=cits[:5],
        )

    # -------------------- Clause Rewriter --------------------
    async def _rewrite_clause(
        self, contract_type: str, jurisdiction: str, role: str,
        clause_text: str, original_identifier: str,
        risk_assessment: RiskAssessment, references_md: str
    ) -> str:
        if not getattr(settings,"rewrite_enabled",True): return ""
        clean_recos=[r for r in (risk_assessment.recommendations or []) if not _is_sufficient(r)]
        if not clean_recos: return ""

        pack=load_pack(contract_type)
        tpl_text=pack.prompts.get("clause_rewriter")
        if not tpl_text: return ""

        original_header,_=_extract_header(clause_text)
        prompt=self._render_prompt(
            tpl_text, contract_type=contract_type, jurisdiction=jurisdiction, role=role,
            clause_text=clause_text, original_header=original_header or original_identifier or "",
            risk_factors=getattr(risk_assessment,"risk_factors",[]),
            recommendations=clean_recos, references_md=references_md,
        )

        async with self.sem:
            try: raw=await asyncio.wait_for(self._generate(prompt), timeout=self.request_timeout)
            except Exception as e: logger.warning(f"[rewrite] failed: {e}"); return ""

        s=(raw or "").strip()
        if s.startswith("```"):
            s=re.sub(r"^```[\w-]*\n","",s); s=re.sub(r"\n```$","",s)
        data=None
        try: data=json.loads(s)
        except Exception:
            m=re.search(r"\{[\s\S]*\}$",s)
            if m:
                try: data=json.loads(m.group(0))
                except Exception: data=None
        if not isinstance(data,dict): return ""

        revised=(data.get("revised") or "").strip()
        if len(revised) < max(20, int(len(clause_text)*float(getattr(settings,"rewrite_min_ratio",0.3)))):
            return ""

        return _enforce_header(clause_text, revised)

    # -------------------- Orchestrate --------------------
    async def analyze(
        self, cn: ContractNormalized, contract_type: str, jurisdiction: str, role: str
    ) -> Tuple[RiskAssessment, List[ClauseAnalysisItem]]:
        pack=load_pack(contract_type)
        tpl_text=pack.prompts["clause_audit"]
        categories=pack.categories()

        clause_items: List[ClauseAnalysisItem]=[]
        all_ra: List[RiskAssessment]=[]

        t0=time.perf_counter()
        for cl in cn.clauses:
            # RAG
            refs=await self.retriever.retrieve_similar_chunks(
                cl.original_text, top_k=settings.retrieval_top_k, reference_only=True,
                contract_type=contract_type, jurisdiction=jurisdiction, clause_type=cl.clause_type)
            refs_md=PromptBuilder.format_references([r.dict() if hasattr(r,"dict") else r for r in refs])

            # 카테고리 병렬 호출
            tasks=[ self._category_audit(tpl_text, contract_type, jurisdiction, role, cat,
                                         cl.original_text, refs_md)
                    for cat in categories ]
            results=await asyncio.gather(*tasks, return_exceptions=False)
            cat_results=[r for r in results if r]

            # 목적 조항 판정 + 병합
            is_purpose=_is_purpose_clause_text(cl.original_text, cl.original_identifier)
            ra=self._merge_clause(cat_results, is_purpose=is_purpose)
            all_ra.append(ra)

            # 개정문/형광펜
            revised_text=await self._rewrite_clause(
                contract_type, jurisdiction, role, cl.original_text, cl.original_identifier, ra, refs_md)
            rev_spans=None
            if revised_text:
                raw_spans=compute_revised_spans(cl.original_text, revised_text)
                filtered=_filter_spans_after_header(revised_text, raw_spans)
                major=_select_major_spans(revised_text, filtered)
                rev_spans=[DiffSpan(**s) for s in major] if major else None

            clause_items.append(ClauseAnalysisItem(
                clause_id=cl.clause_id,
                original_identifier=cl.original_identifier,
                original_text=cl.original_text,
                risk_assessment=ra,
                revised_text=revised_text or None,
                revised_spans=rev_spans,
            ))
            logger.info(f"[audit] clause={cl.original_identifier or cl.clause_id} cats={len(categories)} -> risks={len(cat_results)}")

        # overall (UNKNOWN은 0점 반영)
        if not all_ra:
            overall = RiskAssessment(
            risk_level="LOW",
            risk_score=10.0,
            risk_factors=[],
            recommendations=[],
            explanation="No clauses.",
            citations=[],
            )
        else:
            scores_raw=[(0.0 if ra.risk_level=="UNKNOWN" else float(ra.risk_score)) for ra in all_ra]
            scores=sorted(scores_raw, reverse=True)
            topn_k=int(getattr(settings,"overall_topn",4))
            w_max=float(getattr(settings,"overall_weight_max",0.65))
            w_top=float(getattr(settings,"overall_weight_topn",0.35))
            topn=scores[:max(1,topn_k)]
            mean_topn=sum(topn)/max(1,len(topn))
            overall_score=w_max*scores[0]+w_top*mean_topn
            overall_score=max(0.0, min(100.0, overall_score * float(getattr(settings,"score_damping_overall",0.94))))
            level=_level_from_score(overall_score)

            factors: List[str]=[]
            recs: List[str]=[]
            for ra in sorted(all_ra, key=lambda x:(0.0 if x.risk_level=="UNKNOWN" else x.risk_score), reverse=True):
                for f in ra.risk_factors:
                    if f and f not in factors: factors.append(f)
                for rc in ra.recommendations:
                    if rc and rc not in recs: recs.append(rc)
                if len(factors)>=5 and len(recs)>=5: break

            overall = RiskAssessment(
                risk_level=level,
                risk_score=round(overall_score, 1),
                risk_factors=factors[:5],
                recommendations=recs[:5],
                explanation="Aggregated from clause-level assessments.",
                citations=[],
            )

        logger.info(f"[audit] done clauses={len(cn.clauses)} in {time.perf_counter()-t0:.1f}s")
        return overall, clause_items
