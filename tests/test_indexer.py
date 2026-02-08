"""Tests for src/search/indexer.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.data.schemas import PetRecord
from src.search.indexer import (
    PET_INDEX_MAPPING,
    PET_INDEX_NAME,
    create_index,
    index_pets,
)


class TestPetIndexMapping:
    """Tests for index mapping configuration."""

    def test_has_text_embedding_field(self) -> None:
        """Mapping should define text_embedding as dense_vector."""
        props = PET_INDEX_MAPPING["mappings"]["properties"]
        assert props["text_embedding"]["type"] == "dense_vector"
        assert props["text_embedding"]["dims"] == 512

    def test_has_image_embedding_field(self) -> None:
        """Mapping should define image_embedding as dense_vector."""
        props = PET_INDEX_MAPPING["mappings"]["properties"]
        assert props["image_embedding"]["type"] == "dense_vector"
        assert props["image_embedding"]["dims"] == 512

    def test_cosine_similarity(self) -> None:
        """Both embedding fields should use cosine similarity."""
        props = PET_INDEX_MAPPING["mappings"]["properties"]
        assert props["text_embedding"]["similarity"] == "cosine"
        assert props["image_embedding"]["similarity"] == "cosine"

    def test_knn_index_enabled(self) -> None:
        """Embedding fields should have index: True for kNN."""
        props = PET_INDEX_MAPPING["mappings"]["properties"]
        assert props["text_embedding"]["index"] is True
        assert props["image_embedding"]["index"] is True

    def test_pet_id_is_keyword(self) -> None:
        """pet_id should be a keyword field."""
        props = PET_INDEX_MAPPING["mappings"]["properties"]
        assert props["pet_id"]["type"] == "keyword"

    def test_index_name(self) -> None:
        """Default index name should be 'pets'."""
        assert PET_INDEX_NAME == "pets"


class TestCreateIndex:
    """Tests for index creation."""

    def test_creates_new_index(self, mock_es_client: MagicMock) -> None:
        """Should create index when it doesn't exist."""
        mock_es_client.indices.exists.return_value = False
        create_index(mock_es_client, "test_pets")
        mock_es_client.indices.create.assert_called_once_with(
            index="test_pets", body=PET_INDEX_MAPPING
        )

    def test_deletes_existing_index(self, mock_es_client: MagicMock) -> None:
        """Should delete and recreate if index exists."""
        mock_es_client.indices.exists.return_value = True
        create_index(mock_es_client, "test_pets")
        mock_es_client.indices.delete.assert_called_once_with(index="test_pets")
        mock_es_client.indices.create.assert_called_once()


class TestIndexPets:
    """Tests for bulk document indexing."""

    @patch("src.search.indexer.bulk")
    def test_indexes_documents(
        self,
        mock_bulk: MagicMock,
        mock_es_client: MagicMock,
        sample_pet_record: PetRecord,
        fake_embedding: list[float],
    ) -> None:
        """Should call bulk with correct document count."""
        mock_bulk.return_value = (1, [])

        count = index_pets(
            mock_es_client,
            [sample_pet_record],
            [fake_embedding],
            [fake_embedding],
        )
        assert count == 1
        mock_bulk.assert_called_once()

    @patch("src.search.indexer.bulk")
    def test_document_structure(
        self,
        mock_bulk: MagicMock,
        mock_es_client: MagicMock,
        sample_pet_record: PetRecord,
        fake_embedding: list[float],
    ) -> None:
        """Indexed documents should contain embeddings."""
        mock_bulk.return_value = (1, [])

        index_pets(
            mock_es_client,
            [sample_pet_record],
            [fake_embedding],
            [fake_embedding],
        )

        actions_generator = mock_bulk.call_args[0][1]
        actions = list(actions_generator)
        assert len(actions) == 1

        doc = actions[0]
        assert doc["_index"] == "pets"
        assert doc["_id"] == "pf-12345"
        assert "text_embedding" in doc["_source"]
        assert "image_embedding" in doc["_source"]
        assert len(doc["_source"]["text_embedding"]) == 512

    @patch("src.search.indexer.bulk")
    def test_returns_success_count(
        self,
        mock_bulk: MagicMock,
        mock_es_client: MagicMock,
        sample_pet_record: PetRecord,
        fake_embedding: list[float],
    ) -> None:
        """Should return the count of successfully indexed docs."""
        mock_bulk.return_value = (5, [])

        count = index_pets(
            mock_es_client,
            [sample_pet_record],
            [fake_embedding],
            [fake_embedding],
        )
        assert count == 5
