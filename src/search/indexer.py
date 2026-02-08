"""Elasticsearch index creation and bulk document indexing."""

from __future__ import annotations

import logging

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.data.schemas import PetRecord

logger = logging.getLogger(__name__)

PET_INDEX_NAME = "pets"

PET_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "pet_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "name": {"type": "text", "analyzer": "standard"},
            "species": {"type": "keyword"},
            "breed": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "age_months": {"type": "integer"},
            "gender": {"type": "keyword"},
            "description": {"type": "text", "analyzer": "standard"},
            "image_path": {"type": "keyword", "index": False},
            "metadata": {"type": "object", "enabled": False},
            "text_embedding": {
                "type": "dense_vector",
                "dims": 512,
                "index": True,
                "similarity": "cosine",
            },
            "image_embedding": {
                "type": "dense_vector",
                "dims": 512,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}


def create_index(es: Elasticsearch, index_name: str = PET_INDEX_NAME) -> None:
    """Create the pets index with dense vector mappings.

    Deletes existing index if present and recreates it.

    Args:
        es: Elasticsearch client.
        index_name: Name of the index to create.
    """
    if es.indices.exists(index=index_name):
        logger.info("Deleting existing index '%s'", index_name)
        es.indices.delete(index=index_name)

    es.indices.create(index=index_name, body=PET_INDEX_MAPPING)
    logger.info("Created index '%s' with dense vector mappings", index_name)


def index_pets(
    es: Elasticsearch,
    records: list[PetRecord],
    text_embeddings: list[list[float]],
    image_embeddings: list[list[float]],
    index_name: str = PET_INDEX_NAME,
    batch_size: int = 100,
) -> int:
    """Bulk index pet records with their CLIP embeddings.

    Args:
        es: Elasticsearch client.
        records: List of PetRecord objects.
        text_embeddings: CLIP text embeddings aligned with records.
        image_embeddings: CLIP image embeddings aligned with records.
        index_name: Target index name.
        batch_size: Bulk indexing batch size.

    Returns:
        Number of successfully indexed documents.
    """

    def _generate_actions():
        for record, text_emb, img_emb in zip(
            records, text_embeddings, image_embeddings, strict=True
        ):
            doc = record.model_dump()
            doc["text_embedding"] = text_emb
            doc["image_embedding"] = img_emb
            yield {
                "_index": index_name,
                "_id": record.pet_id,
                "_source": doc,
            }

    success, errors = bulk(es, _generate_actions(), chunk_size=batch_size)

    if errors:
        logger.error("Bulk indexing errors: %s", errors)

    logger.info("Indexed %d documents into '%s'", success, index_name)
    return success
