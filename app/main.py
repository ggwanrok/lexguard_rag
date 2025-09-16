from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.config import settings

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="계약서 RAG 분석 API",
)

# In production, replace "*" with allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "LexGuard RAG API running"}

@app.get("/health")
async def health():
    return {"status": "ok"}
