# 📑 RAG 기반 계약서 리스크 분석 시스템

## 개요
이 프로젝트는 **계약서 조항별 위험 요소 분석 및 개선 제안**을 자동화하기 위해 개발된 **RAG(Retrieval-Augmented Generation) 기반 AI 시스템**입니다.  
OCR을 통해 스캔된 계약서를 텍스트로 변환하고, 벡터 임베딩 + Qdrant DB + LLM 분석 파이프라인을 통해 조항별 리스크를 진단합니다.  

## 주요 기능
- **계약서 OCR 처리**: 스캔본 PDF → 텍스트 추출  
- **조항 분리 및 전처리**: 계약서를 조항 단위로 분리 후 정규화  
- **임베딩 + 벡터 검색**: Qdrant에 저장된 법률 관련 데이터와 유사도 검색  
- **1차 분석 (LLM)**: 각 조항의 리스크 및 개선안 도출  
- **2차 검증 (LLM)**: 원문 대비 결과 검증 및 신뢰도 보정  
- **리스크 시각화**: 결과 확인  
- **PDF 보고서 다운로드**

## 시스템 아키텍처
```mermaid
flowchart TD
    A[OCR] --> B[Chunker]
    B --> C[Embedder]
    C --> D[Qdrant(Vector DB)]
    D --> E[Audit Orchestrator]
    E --> F[LLM Evaluator (1차 분석)]
    F --> G[Clause Rewriter / Risk Validator (2차 검증)]
    G --> H[결과 저장 및 시각화 (Viewer, PDF)]
```

## 폴더 구조
```
rag_contract_audit/
  ├── requirements.txt       # Python 패키지 의존성
  ├── .env                   # 환경 변수 (API 키, DB 설정 등)
  ├── nda_risk_viewer.html   # 분석 결과 뷰어
  ├── rag_contract_audit/    # 핵심 백엔드 모듈
  └── .qdrant_storage/       # Qdrant 로컬 스토리지
```

## 기술 스택
- **Python 3.10+**
- **FastAPI** (REST API)
- **Qdrant** (벡터 데이터베이스)
- **Gemini API**
- **OpenAI GPT / Embeddings API**
- **OCR (Mistral AI)**

## 향후 개선
- 리스크 분석 결과에 점수 기반 필터링
- 계약서 유형별 맞춤형 프롬프트 강화
