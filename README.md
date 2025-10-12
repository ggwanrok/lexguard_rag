# RAG 기반 계약서 리스크 분석 시스템

## 개요
이 프로젝트는 계약서를 조항 단위로 정규화한 뒤, 사전 구축한 레퍼런스(참조 스니펫)와의 유사도 검색(RAG)을 거쳐 LLM으로 위험 신호를 진단하고 개선문을 제안하는 시스템입니다.  
최종 응답은 **조항별 리스크 등급/점수, 권장안, 개정문 및 diff 하이라이트, 전체 요약**을 포함합니다.  

## 아키텍처 개요
- **Normalizer**: 원문을 조항 단위로 분리·정규화.
- **Retriever (Qdrant)**: 임베딩 기반 유사도 검색으로 관련 레퍼런스 스니펫 제공.
- **Audit Orchestrator (Gemini)**: 카테고리별 점검 → 점수/등급 보정 및 병합 → 조항 리스크 산출.
- **Rewriter**: 실질적 권장안이 있을 때, 개정문 생성, diff span 산출.
- **Summarizer**: 전체 요약 및 리포트 텍스트 생성.
- **API (FastAPI)**: JSON/Markdown/Plain 응답, 파일 업로드/생텍스트 분석 지원.

## 주요 기능
- **조항 분리 및 전처리/정규화**
- **Qdrant 기반 벡터 검색(RAG)**: 유사 레퍼런스 스니펫 검색
- **카테고리별 LLM 점검 및 근거 취합**: 트리거/인용/설명 근거 반영
- **점수/등급 산출 및 보정**: 중립 구간(40~60) 당김, 에비던스 기반 가중
- **개정문 생성 + diff 하이라이트**
- **Overall 집계**: 상위 점수 일부 가중 평균 후 감쇠
- **검색 API**: 유사 스니펫 조회

## 기술 스택
- **Python 3.11+**
- **FastAPI** (REST API)
- **Qdrant** (Vector DB)
- **Embeddings**: Hugging Face kakao1513/KURE-legal-ft-v1
- **LLM**: Google **Gemini**
- **Pandas**(CSV ingest), **Jinja2**(프롬프트 템플릿)
