from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .data import AD_TEMPLATES
from .pipeline import (
    translate_to_english_if_needed,
    build_template_embeddings,
    semantic_search,
)

# Build template embeddings at startup
TEMPLATE_EMBEDDINGS = build_template_embeddings(AD_TEMPLATES)

app = FastAPI(title="Multilingual Contextual Search")


class QueryInput(BaseModel):
    query: str
    language: Optional[str] = None  # 'en', 'es', 'auto', etc.
    limit: int = 5


@app.post("/search")
async def search(query_input: QueryInput):
    query_text = translate_to_english_if_needed(
        text=query_input.query,
        language_hint=query_input.language,
    )

    results = semantic_search(
        query_text=query_text,
        templates=AD_TEMPLATES,
        template_embeddings=TEMPLATE_EMBEDDINGS,
        limit=query_input.limit,
    )
    return {"results": results}


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Use POST /search with JSON body or GET /search with query params. See /docs for Swagger UI.",
    }


@app.get("/search")
async def search_get(query: str, language: Optional[str] = None, limit: int = 5):
    query_text = translate_to_english_if_needed(
        text=query,
        language_hint=language,
    )
    results = semantic_search(
        query_text=query_text,
        templates=AD_TEMPLATES,
        template_embeddings=TEMPLATE_EMBEDDINGS,
        limit=limit,
    )
    return {"results": results}