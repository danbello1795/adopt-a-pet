"""FastAPI routes for text search, image search, and health check."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.data.schemas import PetRecord, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Curated breeds for the homepage sample cards (3 dogs, 3 cats)
_FEATURED_BREEDS: list[tuple[str, str]] = [
    ("Golden Retriever", "Dog"),
    ("German Shepherd Dog", "Dog"),
    ("Labrador Retriever", "Dog"),
    ("Persian", "Cat"),
    ("Siamese", "Cat"),
    ("Ragdoll", "Cat"),
]


def _fetch_featured_pets(request: Request) -> tuple[int, list[PetRecord]]:
    """Fetch pet count and one representative pet per featured breed.

    Args:
        request: FastAPI request with app state.

    Returns:
        Tuple of (total pet count, list of featured PetRecord objects).
    """
    es = request.app.state.es_client
    config = request.app.state.config

    try:
        pet_count = es.count(index=config.index_name)["count"]
    except Exception:
        logger.warning("Failed to fetch pet count from ES")
        pet_count = 0

    featured: list[PetRecord] = []
    for breed, species in _FEATURED_BREEDS:
        try:
            resp = es.search(
                index=config.index_name,
                body={
                    "size": 1,
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"species": species}},
                                {"match": {"breed": breed}},
                            ]
                        }
                    },
                    "_source": {
                        "excludes": ["text_embedding", "image_embedding"]
                    },
                },
            )
            hits = resp["hits"]["hits"]
            if hits:
                featured.append(PetRecord(**hits[0]["_source"]))
        except Exception:
            logger.warning("Failed to fetch featured pet for breed=%s", breed)

    return pet_count, featured


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Landing page with text search bar and image upload form."""
    pet_count, featured_pets = _fetch_featured_pets(request)
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "config": request.app.state.config,
            "pet_count": pet_count,
            "featured_pets": featured_pets,
        },
    )


@router.get("/search", response_class=HTMLResponse)
async def text_search(request: Request, q: str = "", top_k: int = 10) -> HTMLResponse:
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
    file: UploadFile = File(...),  # noqa: B008
    top_k: int = Form(10),  # noqa: B008
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
async def api_text_search(request: Request, q: str, top_k: int = 10) -> SearchResponse:
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
