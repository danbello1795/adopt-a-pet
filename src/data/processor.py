"""Process and merge PetFinder and Oxford-IIIT datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.schemas import PetRecord

logger = logging.getLogger(__name__)

SPECIES_MAP = {1: "Dog", 2: "Cat"}
GENDER_MAP = {1: "Male", 2: "Female", 3: "Mixed"}


def process_petfinder(
    petfinder_dir: Path,
    sample_size: int = 1000,
    random_seed: int = 42,
) -> list[PetRecord]:
    """Process PetFinder data into normalized PetRecord objects.

    Loads train.csv, joins with breed/color labels, filters records
    with descriptions and photos, then samples.

    Args:
        petfinder_dir: Path to extracted PetFinder data.
        sample_size: Number of records to sample.
        random_seed: Random seed for reproducibility.

    Returns:
        List of PetRecord objects.
    """
    logger.info("Processing PetFinder dataset from %s", petfinder_dir)

    train_csv = petfinder_dir / "train.csv"
    if not train_csv.exists():
        train_csv = petfinder_dir / "train" / "train.csv"
    train_df = pd.read_csv(train_csv)
    breed_labels = pd.read_csv(petfinder_dir / "breed_labels.csv")
    color_labels = pd.read_csv(petfinder_dir / "color_labels.csv")

    breed_map = dict(
        zip(breed_labels["BreedID"], breed_labels["BreedName"], strict=False)
    )
    color_map = dict(
        zip(color_labels["ColorID"], color_labels["ColorName"], strict=False)
    )

    train_df["BreedName"] = train_df["Breed1"].map(breed_map).fillna("Mixed")
    train_df["ColorName"] = train_df["Color1"].map(color_map).fillna("Unknown")
    train_df["SpeciesName"] = train_df["Type"].map(SPECIES_MAP)
    train_df["GenderName"] = train_df["Gender"].map(GENDER_MAP)

    has_description = train_df["Description"].notna() & (
        train_df["Description"].str.strip() != ""
    )
    has_photo = train_df["PhotoAmt"] > 0
    valid_df = train_df[has_description & has_photo].copy()

    logger.info(
        "PetFinder: %d total, %d with description+photo",
        len(train_df),
        len(valid_df),
    )

    sample_size = min(sample_size, len(valid_df))
    sampled = valid_df.sample(n=sample_size, random_state=random_seed)

    images_dir = _find_images_dir(petfinder_dir)

    records = []
    for _, row in sampled.iterrows():
        image_filename = f"{row['PetID']}-1.jpg"
        image_path = images_dir / image_filename if images_dir else None

        if image_path and not image_path.exists():
            continue

        description = _build_petfinder_description(row)

        # Handle NaN values for name field
        name = row.get("Name", "Unknown")
        if pd.isna(name):
            name = "Unknown"

        records.append(
            PetRecord(
                pet_id=f"pf-{row['PetID']}",
                source="petfinder",
                name=name,
                species=row["SpeciesName"],
                breed=row["BreedName"],
                age_months=int(row["Age"]) if pd.notna(row["Age"]) else None,
                gender=row.get("GenderName"),
                description=description,
                image_path=str(image_path) if image_path else "",
                metadata={
                    "color": row["ColorName"],
                    "fee": int(row["Fee"]) if pd.notna(row.get("Fee")) else 0,
                    "vaccinated": int(row.get("Vaccinated", 0)),
                    "sterilized": int(row.get("Sterilized", 0)),
                    "adoption_speed": int(row.get("AdoptionSpeed", -1)),
                },
            )
        )

    logger.info("PetFinder: produced %d records", len(records))
    return records


def process_oxford(
    oxford_dir: Path,
    sample_size: int = 500,
    random_seed: int = 42,
) -> list[PetRecord]:
    """Process Oxford-IIIT Pet Dataset into normalized PetRecord objects.

    Parses annotations for breed/species mapping, samples images,
    and generates synthetic descriptions.

    Args:
        oxford_dir: Path to extracted Oxford-IIIT data.
        sample_size: Number of records to sample.
        random_seed: Random seed for reproducibility.

    Returns:
        List of PetRecord objects.
    """
    logger.info("Processing Oxford-IIIT dataset from %s", oxford_dir)

    annotations_path = oxford_dir / "annotations" / "list.txt"
    images_dir = oxford_dir / "images"

    annotations = _parse_oxford_annotations(annotations_path)

    valid_records = []
    for entry in annotations:
        image_path = images_dir / f"{entry['filename']}.jpg"
        if image_path.exists():
            valid_records.append({**entry, "image_path": image_path})

    logger.info(
        "Oxford-IIIT: %d annotations, %d with valid images",
        len(annotations),
        len(valid_records),
    )

    sample_size = min(sample_size, len(valid_records))
    df = pd.DataFrame(valid_records)
    sampled = df.sample(n=sample_size, random_state=random_seed)

    records = []
    for _, row in sampled.iterrows():
        breed = row["breed"].replace("_", " ").title()
        species = SPECIES_MAP.get(row["species_id"], "Unknown")
        description = (
            f"A {breed} {species.lower()}. "
            f"This is a photo from the Oxford-IIIT Pet Dataset."
        )

        records.append(
            PetRecord(
                pet_id=f"ox-{row['filename']}",
                source="oxford_iiit",
                name=breed,
                species=species,
                breed=breed,
                description=description,
                image_path=str(row["image_path"]),
            )
        )

    logger.info("Oxford-IIIT: produced %d records", len(records))
    return records


def merge_datasets(
    petfinder_records: list[PetRecord],
    oxford_records: list[PetRecord],
) -> list[PetRecord]:
    """Merge records from both datasets into a unified list.

    Args:
        petfinder_records: Records from PetFinder.
        oxford_records: Records from Oxford-IIIT.

    Returns:
        Combined list of PetRecord objects.
    """
    merged = petfinder_records + oxford_records
    logger.info(
        "Merged datasets: %d PetFinder + %d Oxford = %d total",
        len(petfinder_records),
        len(oxford_records),
        len(merged),
    )
    return merged


def _build_petfinder_description(row: pd.Series) -> str:
    """Build an optimized text description for CLIP encoding.

    Combines structured metadata with the original description,
    keeping within CLIP's 77-token context window.

    Args:
        row: A row from the PetFinder DataFrame.

    Returns:
        Formatted description string.
    """
    parts = []
    name = row.get("Name")
    if name and str(name).strip() and str(name) != "nan":
        parts.append(str(name).strip())

    breed = row.get("BreedName", "Mixed")
    species = row.get("SpeciesName", "pet")
    parts.append(f"a {breed} {species.lower()}")

    age = row.get("Age")
    if pd.notna(age):
        age_int = int(age)
        if age_int < 12:
            parts.append(f"{age_int} months old")
        else:
            parts.append(f"{age_int // 12} years old")

    description = str(row.get("Description", ""))
    if description and description != "nan":
        parts.append(description[:200])

    return ". ".join(parts)


def _parse_oxford_annotations(annotations_path: Path) -> list[dict]:
    """Parse Oxford-IIIT annotations list.txt file.

    The file has comments starting with # and data rows with format:
    Image CLASS-ID SPECIES BREED-ID

    Args:
        annotations_path: Path to list.txt.

    Returns:
        List of dicts with filename, class_id, species_id, breed.
    """
    entries = []
    with open(annotations_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            filename = parts[0]
            class_id = int(parts[1])
            species_id = int(parts[2])

            breed = "_".join(filename.split("_")[:-1])

            entries.append(
                {
                    "filename": filename,
                    "class_id": class_id,
                    "species_id": species_id,
                    "breed": breed,
                }
            )
    return entries


def _find_images_dir(petfinder_dir: Path) -> Path | None:
    """Find the PetFinder images directory.

    Images may be in train_images/ or directly in the data dir.

    Args:
        petfinder_dir: Base PetFinder data directory.

    Returns:
        Path to images directory, or None if not found.
    """
    candidates = [
        petfinder_dir / "train_images",
        petfinder_dir / "train_images" / "train_images",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path

    logger.warning("PetFinder images directory not found in %s", petfinder_dir)
    return None
