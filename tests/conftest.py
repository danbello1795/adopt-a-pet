"""Shared test fixtures for the Adopt-a-Pet test suite."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.schemas import PetRecord, SearchResponse, SearchResult


@pytest.fixture
def sample_pet_record() -> PetRecord:
    """Create a sample PetRecord for testing."""
    return PetRecord(
        pet_id="pf-12345",
        source="petfinder",
        name="Buddy",
        species="Dog",
        breed="Labrador Retriever",
        age_months=24,
        gender="Male",
        description="A friendly Labrador Retriever. 2 years old.",
        image_path="data/petfinder/train_images/12345-1.jpg",
        metadata={"color": "Golden", "fee": 100},
    )


@pytest.fixture
def sample_oxford_record() -> PetRecord:
    """Create a sample Oxford-IIIT PetRecord for testing."""
    return PetRecord(
        pet_id="ox-Persian_123",
        source="oxford_iiit",
        name="Persian",
        species="Cat",
        breed="Persian",
        description="A Persian cat. This is a photo from the Oxford-IIIT Pet Dataset.",
        image_path="data/oxford_pets/images/Persian_123.jpg",
    )


@pytest.fixture
def sample_search_result(sample_pet_record: PetRecord) -> SearchResult:
    """Create a sample SearchResult for testing."""
    return SearchResult(
        pet=sample_pet_record,
        score=0.95,
        explanation=(
            "Match score: 0.950 | Source: petfinder" " | Breed: Labrador Retriever"
        ),
    )


@pytest.fixture
def sample_search_response(
    sample_search_result: SearchResult,
) -> SearchResponse:
    """Create a sample SearchResponse for testing."""
    return SearchResponse(
        query="friendly dog",
        query_type="text",
        results=[sample_search_result],
        total_hits=1,
        search_time_ms=15.2,
    )


@pytest.fixture
def mock_es_client() -> MagicMock:
    """Create a mock Elasticsearch client."""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.indices.exists.return_value = False
    mock.indices.create.return_value = {"acknowledged": True}
    return mock


@pytest.fixture
def fake_embedding() -> list[float]:
    """Create a fake 512-dim embedding vector."""
    return [0.01] * 512


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
