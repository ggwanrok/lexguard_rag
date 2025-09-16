import pytest
from app.services.chunker import ChunkerService

class TestChunkerService:
    """ChunkerService 테스트 클래스"""
    
    def setup_method(self):
        """테스트 메서드 실행 전 설정"""
        self.chunker = ChunkerService()
    
    def test_chunk_text_basic(self):
        """기본 텍스트 청킹 테스트"""
        text = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'chunk_id') for chunk in chunks)
        assert all(hasattr(chunk, 'content') for chunk in chunks)
    
    def test_chunk_by_clauses(self):
        """조항 단위 청킹 테스트"""
        text = """
        제1조 (목적) 이 계약은 서비스 제공에 관한 사항을 정한다.
        제2조 (서비스 내용) 제공자는 다음과 같은 서비스를 제공한다.
        제3조 (계약 기간) 이 계약의 기간은 1년으로 한다.
        """
        
        chunks = self.chunker.chunk_by_clauses(text)
        
        assert len(chunks) >= 3  # 최소 3개 조항
        assert all('clause' in chunk.metadata.get('type', '') for chunk in chunks)
    
    def test_empty_text(self):
        """빈 텍스트 처리 테스트"""
        chunks = self.chunker.chunk_text("")
        assert len(chunks) == 0
    
    def test_single_sentence(self):
        """단일 문장 처리 테스트"""
        text = "단일 문장입니다."
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].content == text 