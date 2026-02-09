"""Pydantic models for data validation and serialization."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PetRecord(BaseModel):
    """Unified pet record from any data source.

    Normalizes records from PetFinder and Oxford-IIIT into
    a common schema for indexing and search.
    """

    pet_id: str = Field(description="Unique identifier, prefixed by source")
    source: str = Field(description="Dataset source: 'petfinder' or 'oxford_iiit'")
    name: str = Field(default="Unknown", description="Pet name if available")
    species: str = Field(description="'Dog' or 'Cat'")
    breed: str = Field(description="Primary breed name")
    age_months: int | None = Field(default=None, description="Age in months")
    gender: str | None = Field(default=None, description="Male, Female, or None")
    description: str = Field(description="Text description for CLIP encoding")
    image_path: str = Field(description="Relative path to primary image file")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional source-specific fields",
    )


class SearchResult(BaseModel):
    """A single search result returned to the UI."""

    pet: PetRecord
    score: float = Field(description="Combined relevance score")
    explanation: str = Field(default="", description="Human-readable match explanation")


class SearchResponse(BaseModel):
    """Full response from a search query."""

    query: str = Field(description="Original search query or '[uploaded image]'")
    query_type: str = Field(description="'text' or 'image'")
    results: list[SearchResult] = Field(default_factory=list)
    listings: list[SearchResult] = Field(
        default_factory=list,
        description="PetFinder adoption listings (text-focused results)",
    )
    images: list[SearchResult] = Field(
        default_factory=list,
        description="Pet image results from all sources",
    )
    total_hits: int = Field(default=0)
    search_time_ms: float = Field(default=0.0)
