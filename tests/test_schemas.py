"""Tests for src/data/schemas.py."""

from __future__ import annotations

import pytest

from src.data.schemas import PetRecord, SearchResponse, SearchResult


class TestPetRecord:
    """Tests for PetRecord model."""

    def test_create_valid_record(self) -> None:
        """Should create record with all required fields."""
        record = PetRecord(
            pet_id="pf-123",
            source="petfinder",
            species="Dog",
            breed="Labrador",
            description="A friendly dog",
            image_path="images/123.jpg",
        )
        assert record.pet_id == "pf-123"
        assert record.source == "petfinder"
        assert record.name == "Unknown"

    def test_default_optional_fields(self) -> None:
        """Optional fields should have proper defaults."""
        record = PetRecord(
            pet_id="ox-1",
            source="oxford_iiit",
            species="Cat",
            breed="Persian",
            description="A cat",
            image_path="images/cat.jpg",
        )
        assert record.name == "Unknown"
        assert record.age_months is None
        assert record.gender is None
        assert record.metadata == {}

    def test_all_fields_populated(self, sample_pet_record: PetRecord) -> None:
        """Should store all provided fields correctly."""
        assert sample_pet_record.pet_id == "pf-12345"
        assert sample_pet_record.source == "petfinder"
        assert sample_pet_record.name == "Buddy"
        assert sample_pet_record.species == "Dog"
        assert sample_pet_record.breed == "Labrador Retriever"
        assert sample_pet_record.age_months == 24
        assert sample_pet_record.gender == "Male"
        assert sample_pet_record.metadata["color"] == "Golden"

    def test_model_dump(self, sample_pet_record: PetRecord) -> None:
        """model_dump should return a dict with all fields."""
        data = sample_pet_record.model_dump()
        assert isinstance(data, dict)
        assert data["pet_id"] == "pf-12345"
        assert data["metadata"]["fee"] == 100

    def test_missing_required_field_raises(self) -> None:
        """Should raise ValidationError when required fields are missing."""
        with pytest.raises(ValueError):
            PetRecord(pet_id="test")  # type: ignore[call-arg]


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_result(self, sample_pet_record: PetRecord) -> None:
        """Should create result with pet and score."""
        result = SearchResult(
            pet=sample_pet_record,
            score=0.85,
            explanation="Test explanation",
        )
        assert result.score == 0.85
        assert result.pet.pet_id == "pf-12345"

    def test_default_explanation(self, sample_pet_record: PetRecord) -> None:
        """Explanation should default to empty string."""
        result = SearchResult(pet=sample_pet_record, score=0.5)
        assert result.explanation == ""


class TestSearchResponse:
    """Tests for SearchResponse model."""

    def test_create_response(self, sample_search_response: SearchResponse) -> None:
        """Should create response with all fields."""
        assert sample_search_response.query == "friendly dog"
        assert sample_search_response.query_type == "text"
        assert len(sample_search_response.results) == 1
        assert sample_search_response.total_hits == 1
        assert sample_search_response.search_time_ms == 15.2

    def test_empty_response(self) -> None:
        """Should handle empty results."""
        response = SearchResponse(
            query="nonexistent",
            query_type="text",
        )
        assert response.results == []
        assert response.total_hits == 0
        assert response.search_time_ms == 0.0

    def test_response_serialization(
        self, sample_search_response: SearchResponse
    ) -> None:
        """Should serialize to JSON-compatible dict."""
        data = sample_search_response.model_dump()
        assert isinstance(data["results"], list)
        assert data["results"][0]["score"] == 0.95
