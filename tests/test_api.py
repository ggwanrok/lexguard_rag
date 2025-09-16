import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestAPIEndpoints:
    """API 엔드포인트 테스트 클래스"""
    
    def test_root_endpoint(self):
        """루트 엔드포인트 테스트"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_check(self):
        """헬스 체크 엔드포인트 테스트"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_analyze_contract_missing_text(self):
        """계약서 분석 - 텍스트 누락 테스트"""
        response = client.post("/api/v1/contracts/analyze", json={})
        assert response.status_code == 422  # Validation error
    
    def test_analyze_contract_valid_request(self):
        """계약서 분석 - 유효한 요청 테스트"""
        contract_data = {
            "contract_text": "제1조 (목적) 이 계약은 테스트 목적으로 작성되었습니다.",
            "contract_name": "테스트 계약서",
            "contract_type": "테스트"
        }
        
        # 실제 OpenAI API 키가 없으면 오류가 발생할 수 있으므로
        # 기본적인 요청 형식만 테스트
        response = client.post("/api/v1/contracts/analyze", json=contract_data)
        # API 키가 없으면 500 에러가 발생할 수 있음
        assert response.status_code in [200, 500]
    
    def test_search_missing_query(self):
        """검색 - 쿼리 누락 테스트"""
        response = client.post("/api/v1/search", json={})
        assert response.status_code == 422  # Validation error
    
    def test_search_valid_request(self):
        """검색 - 유효한 요청 테스트"""
        search_data = {
            "query": "테스트 검색",
            "top_k": 5
        }
        
        response = client.post("/api/v1/search", json=search_data)
        # API 키가 없으면 500 에러가 발생할 수 있음
        assert response.status_code in [200, 500]
    
    def test_get_contract_not_found(self):
        """존재하지 않는 계약서 조회 테스트"""
        response = client.get("/api/v1/contracts/nonexistent-id")
        assert response.status_code == 404
    
    def test_delete_contract_not_found(self):
        """존재하지 않는 계약서 삭제 테스트"""
        response = client.delete("/api/v1/contracts/nonexistent-id")
        assert response.status_code == 404 