"""Tests for src/data/processor.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.processor import (
    _build_petfinder_description,
    _parse_oxford_annotations,
    merge_datasets,
    process_oxford,
    process_petfinder,
)
from src.data.schemas import PetRecord


@pytest.fixture
def petfinder_data_dir(tmp_path: Path) -> Path:
    """Create a fake PetFinder data directory."""
    pf_dir = tmp_path / "petfinder"
    pf_dir.mkdir()

    train_data = pd.DataFrame(
        {
            "PetID": ["A001", "A002", "A003"],
            "Type": [1, 2, 1],
            "Name": ["Rex", "Mimi", "Buddy"],
            "Breed1": [1, 2, 3],
            "Color1": [1, 2, 3],
            "Gender": [1, 2, 1],
            "Age": [12, 6, 24],
            "Description": [
                "A friendly dog",
                "A cute cat",
                "A playful puppy",
            ],
            "PhotoAmt": [1, 1, 1],
            "Fee": [100, 50, 0],
            "Vaccinated": [1, 1, 0],
            "Sterilized": [1, 0, 1],
            "AdoptionSpeed": [2, 1, 3],
        }
    )
    train_data.to_csv(pf_dir / "train.csv", index=False)

    breed_labels = pd.DataFrame(
        {
            "BreedID": [1, 2, 3],
            "BreedName": ["Labrador", "Siamese", "Golden Retriever"],
        }
    )
    breed_labels.to_csv(pf_dir / "breed_labels.csv", index=False)

    color_labels = pd.DataFrame(
        {
            "ColorID": [1, 2, 3],
            "ColorName": ["Black", "White", "Golden"],
        }
    )
    color_labels.to_csv(pf_dir / "color_labels.csv", index=False)

    images_dir = pf_dir / "train_images"
    images_dir.mkdir()
    for pet_id in ["A001", "A002", "A003"]:
        (images_dir / f"{pet_id}-1.jpg").write_bytes(b"\xff\xd8\xff")

    return pf_dir


@pytest.fixture
def oxford_data_dir(tmp_path: Path) -> Path:
    """Create a fake Oxford-IIIT data directory."""
    ox_dir = tmp_path / "oxford_pets"
    ox_dir.mkdir()

    annotations_dir = ox_dir / "annotations"
    annotations_dir.mkdir()

    annotations_content = """# Comment line
# Another comment
Abyssinian_1 1 1 1
Bengal_2 2 1 2
Boxer_3 3 2 3
"""
    (annotations_dir / "list.txt").write_text(annotations_content)

    images_dir = ox_dir / "images"
    images_dir.mkdir()
    for name in ["Abyssinian_1", "Bengal_2", "Boxer_3"]:
        (images_dir / f"{name}.jpg").write_bytes(b"\xff\xd8\xff")

    return ox_dir


class TestProcessPetfinder:
    """Tests for PetFinder processing."""

    def test_produces_pet_records(self, petfinder_data_dir: Path) -> None:
        """Should return a list of PetRecord objects."""
        records = process_petfinder(petfinder_data_dir, sample_size=10)
        assert len(records) > 0
        assert all(isinstance(r, PetRecord) for r in records)

    def test_record_fields(self, petfinder_data_dir: Path) -> None:
        """Records should have correct source and id prefix."""
        records = process_petfinder(petfinder_data_dir, sample_size=10)
        for r in records:
            assert r.source == "petfinder"
            assert r.pet_id.startswith("pf-")

    def test_sample_size_respected(self, petfinder_data_dir: Path) -> None:
        """Should not return more than sample_size records."""
        records = process_petfinder(petfinder_data_dir, sample_size=2)
        assert len(records) <= 2

    def test_breed_mapping(self, petfinder_data_dir: Path) -> None:
        """Breed names should be resolved from breed_labels."""
        records = process_petfinder(petfinder_data_dir, sample_size=10)
        breeds = {r.breed for r in records}
        assert breeds.issubset({"Labrador", "Siamese", "Golden Retriever", "Mixed"})


class TestProcessOxford:
    """Tests for Oxford-IIIT processing."""

    def test_produces_pet_records(self, oxford_data_dir: Path) -> None:
        """Should return a list of PetRecord objects."""
        records = process_oxford(oxford_data_dir, sample_size=10)
        assert len(records) > 0
        assert all(isinstance(r, PetRecord) for r in records)

    def test_record_fields(self, oxford_data_dir: Path) -> None:
        """Records should have correct source and id prefix."""
        records = process_oxford(oxford_data_dir, sample_size=10)
        for r in records:
            assert r.source == "oxford_iiit"
            assert r.pet_id.startswith("ox-")

    def test_species_detection(self, oxford_data_dir: Path) -> None:
        """Species should be correctly identified."""
        records = process_oxford(oxford_data_dir, sample_size=10)
        species_set = {r.species for r in records}
        assert species_set.issubset({"Dog", "Cat", "Unknown"})

    def test_description_generated(self, oxford_data_dir: Path) -> None:
        """Descriptions should be generated from breed/species."""
        records = process_oxford(oxford_data_dir, sample_size=10)
        for r in records:
            assert len(r.description) > 0
            assert "Oxford-IIIT" in r.description


class TestMergeDatasets:
    """Tests for dataset merging."""

    def test_merge_combines_both_lists(
        self,
        sample_pet_record: PetRecord,
        sample_oxford_record: PetRecord,
    ) -> None:
        """Should combine records from both datasets."""
        merged = merge_datasets([sample_pet_record], [sample_oxford_record])
        assert len(merged) == 2

    def test_merge_preserves_order(
        self,
        sample_pet_record: PetRecord,
        sample_oxford_record: PetRecord,
    ) -> None:
        """PetFinder records should come first."""
        merged = merge_datasets([sample_pet_record], [sample_oxford_record])
        assert merged[0].source == "petfinder"
        assert merged[1].source == "oxford_iiit"

    def test_merge_empty_lists(self) -> None:
        """Should handle empty input lists."""
        merged = merge_datasets([], [])
        assert merged == []


class TestBuildPetfinderDescription:
    """Tests for description builder."""

    def test_includes_breed_and_species(self) -> None:
        """Description should contain breed and species."""
        row = pd.Series(
            {
                "Name": "Rex",
                "BreedName": "Labrador",
                "SpeciesName": "Dog",
                "Age": 12,
                "Description": "A friendly dog",
            }
        )
        desc = _build_petfinder_description(row)
        assert "Labrador" in desc
        assert "dog" in desc.lower()

    def test_truncates_long_description(self) -> None:
        """Should truncate description to 200 chars."""
        row = pd.Series(
            {
                "Name": "",
                "BreedName": "Mix",
                "SpeciesName": "Dog",
                "Age": None,
                "Description": "x" * 500,
            }
        )
        desc = _build_petfinder_description(row)
        assert len(desc) < 500

    def test_handles_missing_name(self) -> None:
        """Should work without a name."""
        row = pd.Series(
            {
                "Name": None,
                "BreedName": "Persian",
                "SpeciesName": "Cat",
                "Age": 6,
                "Description": "A cute cat",
            }
        )
        desc = _build_petfinder_description(row)
        assert "Persian" in desc


class TestParseOxfordAnnotations:
    """Tests for Oxford annotations parser."""

    def test_parse_annotations(self, oxford_data_dir: Path) -> None:
        """Should parse annotation lines correctly."""
        annotations = _parse_oxford_annotations(
            oxford_data_dir / "annotations" / "list.txt"
        )
        assert len(annotations) == 3
        assert annotations[0]["filename"] == "Abyssinian_1"
        assert annotations[0]["class_id"] == 1
        assert annotations[0]["species_id"] == 1

    def test_skips_comments(self, oxford_data_dir: Path) -> None:
        """Should skip comment lines starting with #."""
        annotations = _parse_oxford_annotations(
            oxford_data_dir / "annotations" / "list.txt"
        )
        assert all(not a["filename"].startswith("#") for a in annotations)

    def test_breed_extraction(self, oxford_data_dir: Path) -> None:
        """Should extract breed from filename."""
        annotations = _parse_oxford_annotations(
            oxford_data_dir / "annotations" / "list.txt"
        )
        assert annotations[0]["breed"] == "Abyssinian"
        assert annotations[2]["breed"] == "Boxer"
