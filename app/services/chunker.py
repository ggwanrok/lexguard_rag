from typing import List
import re
import uuid
from app.models.schema import ChunkInfo
from app.config import settings

class ChunkerService:
    """계약서 텍스트 청킹 서비스"""

    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    # ---- 조항 기준(가능하면 우선) ----
    def chunk_by_clauses(self, text: str) -> List[ChunkInfo]:
        """
        '제 n 조' 패턴 기준 1차 분할.
        마지막 조항 경계 처리 및 간단한 소항목까지 포함.
        """
        clause_pattern = r'(제\\s*\\d+\\s*조[^\\n]*)'
        parts = re.split(clause_pattern, text)
        if len(parts) <= 1:
            return []

        chunks: List[ChunkInfo] = []
        # parts: ["문서머리", "제1조 제목", "제1조 본문...", ...]
        head = parts[0]
        idx_cursor = len(head)

        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            body = parts[i + 1] if (i + 1) < len(parts) else ""
            content = f"{title}\\n{body}".strip()
            chunk_id = str(uuid.uuid4())
            start_index = text.find(content, idx_cursor)
            if start_index < 0:
                start_index = idx_cursor
            end_index = start_index + len(content)
            idx_cursor = end_index

            chunks.append(
                ChunkInfo(
                    chunk_id=chunk_id,
                    content=content,
                    start_index=start_index,
                    end_index=end_index,
                    metadata={
                        "type": "clause",
                        "clause_number": title,
                        "clause_type": None,  # 필요 시 규칙/모델로 부여
                        "word_count": len(content.split()),
                    },
                )
            )
        return chunks

    # ---- 일반 슬라이딩 윈도우 ----
    def chunk_text(self, text: str) -> List[ChunkInfo]:
        sentences = self._split_into_sentences(text)
        chunks: List[ChunkInfo] = []
        current = ""
        start_index = 0

        for s in sentences:
            candidate = f"{current} {s}".strip() if current else s
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunk_id = str(uuid.uuid4())
                    end_index = start_index + len(current)
                    chunks.append(
                        ChunkInfo(
                            chunk_id=chunk_id,
                            content=current,
                            start_index=start_index,
                            end_index=end_index,
                            metadata={
                                "sentence_count": len([t for t in current.split('.') if t.strip()]),
                                "word_count": len(current.split()),
                            },
                        )
                    )
                    overlap_start = max(0, end_index - self.chunk_overlap)
                    current = (text[overlap_start:end_index] + " " + s).strip()
                    start_index = overlap_start
                else:
                    current = s
                    start_index = text.find(s)

        if current:
            chunk_id = str(uuid.uuid4())
            end_index = start_index + len(current)
            chunks.append(
                ChunkInfo(
                    chunk_id=chunk_id,
                    content=current,
                    start_index=start_index,
                    end_index=end_index,
                    metadata={
                        "sentence_count": len([t for t in current.split('.') if t.strip()]),
                        "word_count": len(current.split()),
                    },
                )
            )
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        pattern = r'(?<=[.!?])\\s+|\\n\\s*\\d+\\.\\s+|\\n\\s*[가-힣]\\.\\s+|\\n\\s*[a-zA-Z]\\.\\s+'
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p and p.strip()]
