"""Download PetFinder and Oxford-IIIT datasets."""

from __future__ import annotations

import json
import logging
import os
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

OXFORD_IMAGES_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
OXFORD_ANNOTATIONS_URL = (
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
)
PETFINDER_COMPETITION = "petfinder-adoption-prediction"
KAGGLE_COMPETITION_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/competitions/data/download-all"
)


def _get_kaggle_key() -> str:
    """Resolve Kaggle API key from env vars or ~/.kaggle/kaggle.json.

    Returns:
        The API key string.

    Raises:
        RuntimeError: If no credentials are found.
    """
    key = os.getenv("KAGGLE_KEY")
    if key:
        return key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        data = json.loads(kaggle_json.read_text())
        if "key" in data:
            return data["key"]

    raise RuntimeError(
        "Kaggle credentials not found. "
        "Set KAGGLE_KEY in your .env file or place kaggle.json in ~/.kaggle/"
    )


def download_petfinder(data_dir: Path) -> Path:
    """Download PetFinder dataset via Kaggle REST API.

    Uses Bearer token auth with the Kaggle API key.
    Reads credentials from KAGGLE_KEY env var or ~/.kaggle/kaggle.json.

    Args:
        data_dir: Base directory for storing datasets.

    Returns:
        Path to extracted PetFinder data directory.

    Raises:
        RuntimeError: If credentials are missing or download fails.
    """
    target = data_dir / "petfinder"
    has_data = (target / "train.csv").exists() or (
        target / "train" / "train.csv"
    ).exists()
    if has_data:
        logger.info("PetFinder data already exists at %s, skipping", target)
        return target

    key = _get_kaggle_key()
    target.mkdir(parents=True, exist_ok=True)

    zip_path = target / f"{PETFINDER_COMPETITION}.zip"
    url = f"{KAGGLE_COMPETITION_DOWNLOAD_URL}/{PETFINDER_COMPETITION}"
    logger.info("Downloading PetFinder dataset from Kaggle API...")

    _download_file(url, zip_path, bearer_token=key)
    _extract_zip(zip_path, target)
    zip_path.unlink()

    logger.info("PetFinder dataset downloaded to %s", target)
    return target


def download_oxford_pets(data_dir: Path) -> Path:
    """Download Oxford-IIIT Pet Dataset via HTTP.

    Downloads images and annotations tar.gz files and extracts them.

    Args:
        data_dir: Base directory for storing datasets.

    Returns:
        Path to extracted Oxford-IIIT data directory.

    Raises:
        RuntimeError: If download fails.
    """
    target = data_dir / "oxford_pets"
    if (target / "images").exists() and (target / "annotations").exists():
        logger.info("Oxford-IIIT data already exists at %s, skipping", target)
        return target

    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Oxford-IIIT Pet Dataset...")

    for url, name in [
        (OXFORD_IMAGES_URL, "images"),
        (OXFORD_ANNOTATIONS_URL, "annotations"),
    ]:
        tar_path = target / f"{name}.tar.gz"
        _download_file(url, tar_path)
        _extract_tar(tar_path, target)
        tar_path.unlink()

    logger.info("Oxford-IIIT dataset downloaded to %s", target)
    return target


def _download_file(
    url: str, dest: Path, *, bearer_token: str | None = None
) -> None:
    """Download a file with progress bar.

    Args:
        url: URL to download from.
        dest: Destination file path.
        bearer_token: Optional Bearer token for authenticated requests.

    Raises:
        RuntimeError: If download fails.
    """
    logger.info("Downloading %s...", dest.name)
    headers = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    try:
        response = requests.get(
            url, stream=True, timeout=300, headers=headers
        )
        response.raise_for_status()
    except requests.RequestException as err:
        raise RuntimeError(f"Failed to download {url}: {err}") from err

    total_size = int(response.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a zip archive.

    Args:
        zip_path: Path to the zip file.
        dest: Destination directory.
    """
    logger.info("Extracting %s...", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def _extract_tar(tar_path: Path, dest: Path) -> None:
    """Extract a tar.gz archive.

    Args:
        tar_path: Path to the tar.gz file.
        dest: Destination directory.
    """
    logger.info("Extracting %s...", tar_path.name)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest, filter="data")
