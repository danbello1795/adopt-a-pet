"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.config import get_config
from src.embeddings.clip_encoder import CLIPEncoder
from src.search.es_client import create_es_client
from src.search.searcher import PetSearcher


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared resources on startup, clean up on shutdown.

    Creates Elasticsearch client on startup. CLIP encoder and PetSearcher
    are lazy-loaded on first search to reduce cold start time.
    """
    config = get_config()

    app.state.config = config
    app.state.es_client = create_es_client(
        url=config.elasticsearch_url,
        cloud_id=config.elasticsearch_cloud_id,
        api_key=config.elasticsearch_api_key,
    )
    # Lazy-load CLIP encoder to speed up cold starts
    app.state.clip_encoder = None
    app.state.searcher = None

    yield

    app.state.es_client.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Adopt-a-Pet Search",
        description="Cross-modal search for pet adoption using CLIP + Elasticsearch",
        version="0.1.0",
        lifespan=lifespan,
    )

    from src.api.routes import router

    app.include_router(router)

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    config = get_config()
    data_dir = config.data_dir

    petfinder_images = data_dir / "petfinder" / "train_images"
    if petfinder_images.exists():
        app.mount(
            "/images/petfinder",
            StaticFiles(directory=str(petfinder_images)),
            name="petfinder_images",
        )

    oxford_images = data_dir / "oxford_pets" / "images"
    if oxford_images.exists():
        app.mount(
            "/images/oxford",
            StaticFiles(directory=str(oxford_images)),
            name="oxford_images",
        )

    return app
