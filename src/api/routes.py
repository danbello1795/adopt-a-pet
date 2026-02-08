"""FastAPI routes for text search, image search, and health check."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.data.schemas import SearchResponse

router = APIRouter()
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Landing page with text search bar and image upload form."""
    return templates.TemplateResponse(
        "home.html", {"request": request}
    )


@router.get("/search", response_class=HTMLResponse)
async def text_search(
    request: Request, q: str = "", top_k: int = 20
) -> HTMLResponse:
    """Search pets by text query and render results page.

    Args:
        request: FastAPI request object.
        q: Text search query.
        top_k: Number of results to return.

    Returns:
        HTML response with search results.
    """
    if not q.strip():
        return templates.TemplateResponse(
            "home.html",
            {"request": request, "error": "Please enter a search query."},
        )

    searcher = request.app.state.searcher
    response = searcher.search_by_text(q.strip(), top_k=top_k)

    return templates.TemplateResponse(
        "results.html",
        {"request": request, "response": response, "query": q},
    )


@router.post("/search/image", response_class=HTMLResponse)
async def image_search(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Form(20),
) -> HTMLResponse:
    """Search pets by uploaded image and render results page.

    Args:
        request: FastAPI request object.
        file: Uploaded image file.
        top_k: Number of results to return.

    Returns:
        HTML response with search results.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    searcher = request.app.state.searcher
    response = searcher.search_by_image(image, top_k=top_k)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "response": response,
            "query": f"[Image: {file.filename}]",
        },
    )


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Health check endpoint for monitoring.

    Returns:
        Dict with system health status.
    """
    es = request.app.state.es_client
    es_healthy = es.ping()
    return {
        "status": "healthy" if es_healthy else "degraded",
        "elasticsearch": "connected" if es_healthy else "disconnected",
    }


@router.get("/api/search", response_model=SearchResponse)
async def api_text_search(
    request: Request, q: str, top_k: int = 20
) -> SearchResponse:
    """JSON API endpoint for text search.

    Args:
        request: FastAPI request object.
        q: Text search query.
        top_k: Number of results.

    Returns:
        SearchResponse as JSON.
    """
    searcher = request.app.state.searcher
    return searcher.search_by_text(q, top_k=top_k)
