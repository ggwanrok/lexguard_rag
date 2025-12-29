import re
import uuid
from typing import List, Optional
from app.models.schema import ClauseNormalized, ContractNormalized
from app.utils.logger import logger

ARTICLE_RE = re.compile(r"(?m)^\s*((제\s*\d+\s*조)|(Section\s*\d+(\.\d+)*))\s*(\(([^)]*)\))?")
PARA_RE = re.compile(r"(제\\s*\\d+\\s*항)")
ITEM_RE = re.compile(r"^\\s*(\\d+\\.|[가-힣]\\.\\s*|[a-zA-Z]\\.\\s*)", re.MULTILINE)

class NormalizerService:
    """자유 텍스트를 계약서 스키마(조항/항/소항)로 표준화. 원문 위치(start/end) 보존."""

    def _clean(self, s: str) -> str:
        # 불필요 공백/줄바꿈 정리
        return re.sub(r"\\s+", " ", s).strip()

    def normalize(self, raw_text: str, contract_id: Optional[str], meta: dict) -> ContractNormalized:
        text = raw_text or ""
        clauses: List[ClauseNormalized] = []

        # 1) 기사(조) 단위로 1차 분리
        article_spans = []
        for m in ARTICLE_RE.finditer(text):
            article_spans.append((m.start(), m.end(), m.group(1), m.group(2)))  # start, end_of_header, header, title

        if not article_spans:
            # 조 항목을 못 찾으면 전체를 하나의 조항으로
            cid = "article_1"
            clauses.append(
                ClauseNormalized(
                    clause_id=cid,
                    original_identifier="본문",
                    original_text=text,
                    analysis_text=self._clean(text),
                    start_index=0,
                    end_index=len(text),
                    clause_type="article",
                    article_number=1,
                    title=None,
                )
            )
        else:
            # 각 기사 블럭 추출
            for i, (s, e, header, title) in enumerate(article_spans):
                block_start = s
                block_end = article_spans[i + 1][0] if i + 1 < len(article_spans) else len(text)
                block = text[block_start:block_end]

                # 기사 번호 추출
                num_m = re.search(r"제\\s*(\\d+)\\s*조", header)
                art_no = int(num_m.group(1)) if num_m else (i + 1)
                art_title = title

                # 2) 항 단위 분리
                para_spans = list(PARA_RE.finditer(block))
                if not para_spans:
                    # 항이 없으면 기사 자체를 하나로
                    cid = f"article_{art_no}"
                    clauses.append(
                        ClauseNormalized(
                            clause_id=cid,
                            original_identifier=header.strip(),
                            original_text=block.strip(),
                            analysis_text=self._clean(block),
                            start_index=block_start,
                            end_index=block_end,
                            clause_type="article",
                            article_number=art_no,
                            title=art_title,
                        )
                    )
                else:
                    # 항 블럭들
                    for j, pm in enumerate(para_spans):
                        p_start = pm.start()
                        p_end = para_spans[j + 1].start() if j + 1 < len(para_spans) else len(block)
                        p_block = block[p_start:p_end]

                        # 항 번호
                        pnum_m = re.search(r"제\\s*(\\d+)\\s*항", pm.group(1))
                        pno = int(pnum_m.group(1)) if pnum_m else (j + 1)

                        cid = f"article_{art_no}_paragraph_{pno}"
                        clauses.append(
                            ClauseNormalized(
                                clause_id=cid,
                                original_identifier=f"제{art_no}조 제{pno}항",
                                original_text=p_block.strip(),
                                analysis_text=self._clean(p_block),
                                start_index=block_start + p_start,
                                end_index=block_start + p_end,
                                clause_type="paragraph",
                                article_number=art_no,
                                paragraph_number=pno,
                                title=art_title if j == 0 else None,
                            )
                        )

        cn = ContractNormalized(
            contract_id=contract_id or str(uuid.uuid4()),
            contract_name=meta.get("contract_name"),
            contract_type=meta.get("contract_type"),
            issuer=meta.get("issuer"),
            jurisdiction=meta.get("jurisdiction"),
            language=meta.get("language"),
            subject=meta.get("subject"),
            version=meta.get("version"),
            source_uri=meta.get("source_uri"),
            is_reference=meta.get("is_reference", False),
            clauses=clauses,
        )
        logger.info(f"Normalized contract: {cn.contract_id}, clauses={len(cn.clauses)}")
        return cn
