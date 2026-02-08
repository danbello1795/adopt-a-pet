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

    def search_by_text(
        self, query: str, top_k: int = 20
    ) -> SearchResponse:
        """Search pets by natural language text query.

        Encodes the query with CLIP and searches both text and image
        embedding fields in Elasticsearch.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            SearchResponse with ranked results.
        """
        start = time.monotonic()

        query_vector = self.encoder.encode_text([query])[0]
        body = _build_text_search_query(query_vector, k=top_k)
        response = self.es.search(index=self.index_name, body=body)

        results = _parse_hits(response["hits"]["hits"])
        elapsed_ms = (time.monotonic() - start) * 1000

        return SearchResponse(
            query=query,
            query_type="text",
            results=results,
            total_hits=len(results),
            search_time_ms=round(elapsed_ms, 1),
        )

    def search_by_image(
        self, image: Image.Image, top_k: int = 20
    ) -> SearchResponse:
        """Search pets by uploaded image.

        Encodes the image with CLIP and searches primarily the image
        embedding field, with secondary text embedding matching.

        Args:
            image: PIL Image uploaded by user.
            top_k: Number of results to return.

        Returns:
            SearchResponse with ranked results.
        """
        start = time.monotonic()

        query_vector = self.encoder.encode_single_image(image)
        body = _build_image_search_query(query_vector, k=top_k)
        response = self.es.search(index=self.index_name, body=body)

        results = _parse_hits(response["hits"]["hits"])
        elapsed_ms = (time.monotonic() - start) * 1000

        return SearchResponse(
            query="[uploaded image]",
            query_type="image",
            results=results,
            total_hits=len(results),
            search_time_ms=round(elapsed_ms, 1),
        )


def _build_text_search_query(
    query_vector: list[float],
    k: int = 20,
    num_candidates: int = 100,
) -> dict:
    """Build ES kNN query for text-based search.

    Searches both text_embedding (higher boost) and image_embedding
    fields, combining results via linear score combination.

    Args:
        query_vector: 512-dim CLIP text embedding.
        k: Number of nearest neighbors per field.
        num_candidates: Candidate pool size for approximate kNN.

    Returns:
        Elasticsearch query body.
    """
    return {
        "size": k,
        "knn": [
            {
                "field": "text_embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
                "boost": 1.5,
            },
            {
                "field": "image_embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
                "boost": 1.0,
            },
        ],
    }


def _build_image_search_query(
    query_vector: list[float],
    k: int = 20,
    num_candidates: int = 100,
) -> dict:
    """Build ES kNN query for image-based search.

    Prioritizes image_embedding field with higher boost.

    Args:
        query_vector: 512-dim CLIP image embedding.
        k: Number of nearest neighbors per field.
        num_candidates: Candidate pool size for approximate kNN.

    Returns:
        Elasticsearch query body.
    """
    return {
        "size": k,
        "knn": [
            {
                "field": "image_embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
                "boost": 2.0,
            },
            {
                "field": "text_embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
                "boost": 0.5,
            },
        ],
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
        pet_data = {
            k: v for k, v in source.items() if k not in exclude_fields
        }
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
