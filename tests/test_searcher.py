"""Tests for src/search/searcher.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.data.schemas import PetRecord, SearchResponse
from src.search.searcher import (
    PetSearcher,
    _build_image_search_query,
    _build_text_search_query,
    _generate_explanation,
    _parse_hits,
)


@pytest.fixture
def mock_searcher() -> PetSearcher:
    """Create a PetSearcher with mocked dependencies."""
    mock_es = MagicMock()
    mock_encoder = MagicMock()
    mock_encoder.encode_text.return_value = [[0.01] * 512]
    mock_encoder.encode_single_image.return_value = [0.01] * 512
    return PetSearcher(mock_es, mock_encoder, index_name="test_pets")


@pytest.fixture
def es_hit() -> dict:
    """Create a sample Elasticsearch hit document."""
    return {
        "_score": 0.95,
        "_source": {
            "pet_id": "pf-100",
            "source": "petfinder",
            "name": "Luna",
            "species": "Cat",
            "breed": "Siamese",
            "age_months": 18,
            "gender": "Female",
            "description": "A beautiful Siamese cat",
            "image_path": "data/images/100-1.jpg",
            "metadata": {},
            "text_embedding": [0.1] * 512,
            "image_embedding": [0.2] * 512,
        },
    }


class TestBuildTextSearchQuery:
    """Tests for text search query builder."""

    def test_query_structure(self, fake_embedding: list[float]) -> None:
        """Should build valid kNN query structure."""
        query = _build_text_search_query(fake_embedding, k=10)
        assert "size" in query
        assert "knn" in query
        assert query["size"] == 10

    def test_dual_knn_fields(self, fake_embedding: list[float]) -> None:
        """Should search both text and image embedding fields."""
        query = _build_text_search_query(fake_embedding)
        fields = [knn["field"] for knn in query["knn"]]
        assert "text_embedding" in fields
        assert "image_embedding" in fields

    def test_text_embedding_higher_boost(self, fake_embedding: list[float]) -> None:
        """Text embedding should have higher boost for text queries."""
        query = _build_text_search_query(fake_embedding)
        boosts = {knn["field"]: knn["boost"] for knn in query["knn"]}
        assert boosts["text_embedding"] > boosts["image_embedding"]

    def test_num_candidates(self, fake_embedding: list[float]) -> None:
        """Should set num_candidates for approximate kNN."""
        query = _build_text_search_query(fake_embedding, num_candidates=200)
        for knn in query["knn"]:
            assert knn["num_candidates"] == 200


class TestBuildImageSearchQuery:
    """Tests for image search query builder."""

    def test_image_embedding_higher_boost(self, fake_embedding: list[float]) -> None:
        """Image embedding should have higher boost for image queries."""
        query = _build_image_search_query(fake_embedding)
        boosts = {knn["field"]: knn["boost"] for knn in query["knn"]}
        assert boosts["image_embedding"] > boosts["text_embedding"]

    def test_query_vector_passed(self, fake_embedding: list[float]) -> None:
        """Query vector should be included in each kNN clause."""
        query = _build_image_search_query(fake_embedding)
        for knn in query["knn"]:
            assert knn["query_vector"] == fake_embedding


class TestParseHits:
    """Tests for Elasticsearch hit parsing."""

    def test_parse_single_hit(self, es_hit: dict) -> None:
        """Should parse a hit into a SearchResult."""
        results = _parse_hits([es_hit])
        assert len(results) == 1
        assert results[0].pet.pet_id == "pf-100"
        assert results[0].score == 0.95

    def test_excludes_embedding_fields(self, es_hit: dict) -> None:
        """Should not include embedding vectors in pet data."""
        results = _parse_hits([es_hit])
        pet_data = results[0].pet.model_dump()
        assert "text_embedding" not in pet_data
        assert "image_embedding" not in pet_data

    def test_parse_empty_hits(self) -> None:
        """Should return empty list for no hits."""
        results = _parse_hits([])
        assert results == []


class TestGenerateExplanation:
    """Tests for explanation generation."""

    def test_includes_score(self, sample_pet_record: PetRecord) -> None:
        """Explanation should include the match score."""
        explanation = _generate_explanation(sample_pet_record, 0.85)
        assert "0.850" in explanation

    def test_includes_source(self, sample_pet_record: PetRecord) -> None:
        """Explanation should include the data source."""
        explanation = _generate_explanation(sample_pet_record, 0.5)
        assert "petfinder" in explanation

    def test_includes_breed(self, sample_pet_record: PetRecord) -> None:
        """Explanation should include the breed."""
        explanation = _generate_explanation(sample_pet_record, 0.5)
        assert "Labrador Retriever" in explanation

    def test_includes_species_if_present(self, sample_pet_record: PetRecord) -> None:
        """Explanation should include species when available."""
        explanation = _generate_explanation(sample_pet_record, 0.5)
        assert "Dog" in explanation


class TestPetSearcher:
    """Tests for PetSearcher search methods."""

    def test_search_by_text_returns_response(
        self, mock_searcher: PetSearcher, es_hit: dict
    ) -> None:
        """Text search should return a SearchResponse."""
        mock_searcher.es.search.return_value = {"hits": {"hits": [es_hit]}}
        response = mock_searcher.search_by_text("friendly cat")
        assert isinstance(response, SearchResponse)
        assert response.query_type == "text"
        assert response.query == "friendly cat"

    def test_search_by_text_calls_encoder(
        self, mock_searcher: PetSearcher, es_hit: dict
    ) -> None:
        """Text search should encode the query with CLIP."""
        mock_searcher.es.search.return_value = {"hits": {"hits": [es_hit]}}
        mock_searcher.search_by_text("playful puppy")
        mock_searcher.encoder.encode_text.assert_called_once_with(["playful puppy"])

    def test_search_by_image_returns_response(
        self, mock_searcher: PetSearcher, es_hit: dict
    ) -> None:
        """Image search should return a SearchResponse."""
        mock_searcher.es.search.return_value = {"hits": {"hits": [es_hit]}}
        image = Image.new("RGB", (224, 224))
        response = mock_searcher.search_by_image(image)
        assert isinstance(response, SearchResponse)
        assert response.query_type == "image"
        assert response.query == "[uploaded image]"

    def test_search_by_image_calls_encoder(
        self, mock_searcher: PetSearcher, es_hit: dict
    ) -> None:
        """Image search should encode the image with CLIP."""
        mock_searcher.es.search.return_value = {"hits": {"hits": [es_hit]}}
        image = Image.new("RGB", (224, 224))
        mock_searcher.search_by_image(image)
        mock_searcher.encoder.encode_single_image.assert_called_once_with(image)

    def test_search_measures_time(
        self, mock_searcher: PetSearcher, es_hit: dict
    ) -> None:
        """Search response should include timing."""
        mock_searcher.es.search.return_value = {"hits": {"hits": [es_hit]}}
        response = mock_searcher.search_by_text("test")
        assert response.search_time_ms >= 0
