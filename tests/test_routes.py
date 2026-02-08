"""Tests for src/api/routes.py."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.api.routes import router
from src.data.schemas import PetRecord, SearchResponse, SearchResult


@pytest.fixture
def mock_app() -> FastAPI:
    """Create a FastAPI app with mocked state for testing."""
    app = FastAPI()
    app.include_router(router)

    pet = PetRecord(
        pet_id="pf-1",
        source="petfinder",
        species="Dog",
        breed="Labrador",
        description="A friendly dog",
        image_path="images/1.jpg",
    )
    mock_response = SearchResponse(
        query="friendly dog",
        query_type="text",
        results=[SearchResult(pet=pet, score=0.9, explanation="Match: 0.9")],
        total_hits=1,
        search_time_ms=10.5,
    )

    mock_searcher = MagicMock()
    mock_searcher.search_by_text.return_value = mock_response
    mock_searcher.search_by_image.return_value = SearchResponse(
        query="[uploaded image]",
        query_type="image",
        results=[SearchResult(pet=pet, score=0.85, explanation="Match: 0.85")],
        total_hits=1,
        search_time_ms=20.0,
    )

    mock_es = MagicMock()
    mock_es.ping.return_value = True

    app.state.searcher = mock_searcher
    app.state.es_client = mock_es
    app.state.config = MagicMock()

    return app


@pytest.fixture
def client(mock_app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(mock_app)


class TestHomeRoute:
    """Tests for the home page."""

    def test_home_returns_200(self, client: TestClient) -> None:
        """Home page should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_home_contains_search_form(self, client: TestClient) -> None:
        """Home page should contain a search form."""
        response = client.get("/")
        assert "search" in response.text.lower()


class TestTextSearchRoute:
    """Tests for text search endpoint."""

    def test_search_with_query(self, client: TestClient) -> None:
        """Should return results for a valid query."""
        response = client.get("/search?q=friendly+dog")
        assert response.status_code == 200

    def test_search_empty_query_shows_error(self, client: TestClient) -> None:
        """Should show error for empty query."""
        response = client.get("/search?q=")
        assert response.status_code == 200
        assert "enter a search query" in response.text.lower()

    def test_search_whitespace_query(self, client: TestClient) -> None:
        """Should treat whitespace-only query as empty."""
        response = client.get("/search?q=   ")
        assert response.status_code == 200


class TestImageSearchRoute:
    """Tests for image upload search endpoint."""

    def test_image_search(self, client: TestClient) -> None:
        """Should accept image upload and return results."""
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        assert response.status_code == 200


class TestHealthRoute:
    """Tests for health check endpoint."""

    def test_health_check_healthy(self, client: TestClient) -> None:
        """Should return healthy status when ES is connected."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["elasticsearch"] == "connected"

    def test_health_check_degraded(self, mock_app: FastAPI) -> None:
        """Should return degraded when ES is down."""
        mock_app.state.es_client.ping.return_value = False
        client = TestClient(mock_app)
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "degraded"


class TestApiSearchRoute:
    """Tests for JSON API search endpoint."""

    def test_api_search_returns_json(self, client: TestClient) -> None:
        """Should return JSON response."""
        response = client.get("/api/search?q=cat")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert data["query_type"] == "text"

    def test_api_search_result_structure(self, client: TestClient) -> None:
        """JSON response should have proper structure."""
        response = client.get("/api/search?q=dog")
        data = response.json()
        assert len(data["results"]) > 0
        result = data["results"][0]
        assert "pet" in result
        assert "score" in result
        assert result["pet"]["pet_id"] == "pf-1"
