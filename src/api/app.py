"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
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

    Creates Elasticsearch client, CLIP encoder, and PetSearcher
    instances that are shared across all requests.
    """
    config = get_config()

    app.state.config = config
    app.state.es_client = create_es_client(config.elasticsearch_url)
    app.state.clip_encoder = CLIPEncoder(
        model_name=config.clip_model_name,
        pretrained=config.clip_pretrained,
    )
    app.state.searcher = PetSearcher(
        es_client=app.state.es_client,
        clip_encoder=app.state.clip_encoder,
        index_name=config.index_name,
    )

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

    from src.api.routes import router

    app.include_router(router)

    return app
