"""Cross-modal pet search engine using Elasticsearch kNN."""

from __future__ import annotations

import logging
import time

from elasticsearch import Elasticsearch
from PIL import Image

from src.data.schemas import PetRecord, SearchResponse, SearchResult
from src.embeddings.clip_encoder import CLIPEncoder

logger = logging.getLogger(__name__)


class PetSearcher:
    """Search engine for cross-modal pet search.

    Supports text-to-pet and image-to-pet search using CLIP embeddings
    and Elasticsearch kNN dense vector queries.

    Args:
        es_client: Connected Elasticsearch client.
        clip_encoder: Initialized CLIPEncoder instance.
        index_name: Elasticsearch index to search.
    """

    def __init__(
        self,
        es_client: Elasticsearch,
        clip_encoder: CLIPEncoder,
        index_name: str = "pets",
    ) -> None:
        self.es = es_client
        self.encoder = clip_encoder
        self.index_name = index_name

    def search_by_text(self, query: str, top_k: int = 10) -> SearchResponse:
        """Search pets by natural language text query.

        Returns adoption listings (PetFinder) and image results (both sources)
        as separate collections for the UI to render in distinct sections.
        Runs source-filtered queries to guarantee representation from both
        PetFinder and Oxford-IIIT datasets.

        Args:
            query: Natural language search query.
            top_k: Number of results to return per section.

        Returns:
            SearchResponse with listings and images populated.
        """
        start = time.monotonic()
        query_vector = self.encoder.encode_text([query])[0]

        listings_body = _build_knn_query(
            query_vector,
            primary_field="text_embedding",
            primary_boost=1.5,
            secondary_field="image_embedding",
            secondary_boost=1.0,
            k=top_k,
            source_filter="petfinder",
        )
        images = _fetch_mixed_images(
            self.es,
            self.index_name,
            query_vector,
            primary_field="text_embedding",
            primary_boost=1.5,
            secondary_field="image_embedding",
            secondary_boost=1.0,
            total_k=top_k,
        )

        listings_resp = self.es.search(index=self.index_name, body=listings_body)
        listings = _parse_hits(listings_resp["hits"]["hits"])

        elapsed_ms = (time.monotonic() - start) * 1000

        return SearchResponse(
            query=query,
            query_type="text",
            results=images,
            listings=listings,
            images=images,
            total_hits=len(listings) + len(images),
            search_time_ms=round(elapsed_ms, 1),
        )

    def search_by_image(self, image: Image.Image, top_k: int = 10) -> SearchResponse:
        """Search pets by uploaded image.

        Returns similar images (both sources) and adoption listings (PetFinder)
        with explanations of why each listing matches the uploaded image.

        Args:
            image: PIL Image uploaded by user.
            top_k: Number of results to return per section.

        Returns:
            SearchResponse with listings and images populated.
        """
        start = time.monotonic()
        query_vector = self.encoder.encode_single_image(image)

        images = _fetch_mixed_images(
            self.es,
            self.index_name,
            query_vector,
            primary_field="image_embedding",
            primary_boost=2.0,
            secondary_field="text_embedding",
            secondary_boost=0.5,
            total_k=top_k,
        )
        listings_body = _build_knn_query(
            query_vector,
            primary_field="image_embedding",
            primary_boost=2.0,
            secondary_field="text_embedding",
            secondary_boost=0.5,
            k=top_k,
            source_filter="petfinder",
        )

        listings_resp = self.es.search(index=self.index_name, body=listings_body)
        listings = _parse_hits(listings_resp["hits"]["hits"])

        elapsed_ms = (time.monotonic() - start) * 1000

        return SearchResponse(
            query="[uploaded image]",
            query_type="image",
            results=images,
            listings=listings,
            images=images,
            total_hits=len(listings) + len(images),
            search_time_ms=round(elapsed_ms, 1),
        )


def _fetch_mixed_images(
    es: Elasticsearch,
    index_name: str,
    query_vector: list[float],
    primary_field: str,
    primary_boost: float,
    secondary_field: str,
    secondary_boost: float,
    total_k: int = 10,
    num_candidates: int = 100,
) -> list[SearchResult]:
    """Fetch image results from both PetFinder and Oxford sources.

    Runs separate source-filtered kNN queries and merges the results,
    guaranteeing representation from both datasets. Allocates roughly
    60% of slots to PetFinder and 40% to Oxford, then fills any
    remaining slots from the other source.

    Args:
        es: Elasticsearch client.
        index_name: Index to search.
        query_vector: 512-dim CLIP embedding.
        primary_field: Main embedding field to search.
        primary_boost: Boost for primary field.
        secondary_field: Secondary embedding field.
        secondary_boost: Boost for secondary field.
        total_k: Total images to return.
        num_candidates: Candidate pool size for approximate kNN.

    Returns:
        Merged list of SearchResult objects sorted by score.
    """
    pf_k = max(total_k * 3 // 5, 1)
    ox_k = max(total_k - pf_k, 1)

    pf_body = _build_knn_query(
        query_vector, primary_field, primary_boost,
        secondary_field, secondary_boost,
        k=pf_k, num_candidates=num_candidates,
        source_filter="petfinder",
    )
    ox_body = _build_knn_query(
        query_vector, primary_field, primary_boost,
        secondary_field, secondary_boost,
        k=ox_k, num_candidates=num_candidates,
        source_filter="oxford_iiit",
    )

    pf_resp = es.search(index=index_name, body=pf_body)
    ox_resp = es.search(index=index_name, body=ox_body)

    pf_results = _parse_hits(pf_resp["hits"]["hits"])
    ox_results = _parse_hits(ox_resp["hits"]["hits"])

    seen_ids: set[str] = set()
    merged: list[SearchResult] = []

    for result in sorted(pf_results + ox_results, key=lambda r: r.score, reverse=True):
        if result.pet.pet_id not in seen_ids:
            seen_ids.add(result.pet.pet_id)
            merged.append(result)
        if len(merged) >= total_k:
            break

    return merged


def _build_knn_query(
    query_vector: list[float],
    primary_field: str,
    primary_boost: float,
    secondary_field: str,
    secondary_boost: float,
    k: int = 20,
    num_candidates: int = 100,
    source_filter: str | None = None,
) -> dict:
    """Build an ES kNN query with optional source filtering.

    Args:
        query_vector: 512-dim CLIP embedding.
        primary_field: Main embedding field to search.
        primary_boost: Boost for primary field.
        secondary_field: Secondary embedding field.
        secondary_boost: Boost for secondary field.
        k: Number of nearest neighbors per field.
        num_candidates: Candidate pool size for approximate kNN.
        source_filter: If set, restrict results to this source value.

    Returns:
        Elasticsearch query body.
    """
    knn_clause_primary: dict = {
        "field": primary_field,
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
        "boost": primary_boost,
    }
    knn_clause_secondary: dict = {
        "field": secondary_field,
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
        "boost": secondary_boost,
    }

    if source_filter:
        term_filter = {"term": {"source": source_filter}}
        knn_clause_primary["filter"] = term_filter
        knn_clause_secondary["filter"] = term_filter

    return {
        "size": k,
        "knn": [knn_clause_primary, knn_clause_secondary],
    }


def _parse_hits(hits: list[dict]) -> list[SearchResult]:
    """Convert Elasticsearch hits to SearchResult objects.

    Filters out embedding fields and builds human-readable explanations.

    Args:
        hits: Raw Elasticsearch hit documents.

    Returns:
        List of SearchResult objects.
    """
    results = []
    exclude_fields = {"text_embedding", "image_embedding"}

    for hit in hits:
        source = hit["_source"]
        pet_data = {k: v for k, v in source.items() if k not in exclude_fields}
        pet = PetRecord(**pet_data)
        explanation = _generate_explanation(pet, hit["_score"])

        results.append(
            SearchResult(
                pet=pet,
                score=hit["_score"],
                explanation=explanation,
            )
        )

    return results


def _generate_explanation(pet: PetRecord, score: float) -> str:
    """Generate human-readable explanation for a search result.

    Args:
        pet: The matched pet record.
        score: Elasticsearch relevance score.

    Returns:
        Explanation string.
    """
    parts = [f"Match score: {score:.3f}"]
    parts.append(f"Source: {pet.source}")
    parts.append(f"Breed: {pet.breed}")
    if pet.species:
        parts.append(f"Species: {pet.species}")
    return " | ".join(parts)
