"""Download PetFinder and Oxford-IIIT datasets."""

from __future__ import annotations

import logging
import subprocess
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

OXFORD_IMAGES_URL = (
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
)
OXFORD_ANNOTATIONS_URL = (
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
)
PETFINDER_COMPETITION = "petfinder-adoption-prediction"


def download_petfinder(data_dir: Path) -> Path:
    """Download PetFinder dataset via Kaggle API.

    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables
    or ~/.kaggle/kaggle.json to be configured.

    Args:
        data_dir: Base directory for storing datasets.

    Returns:
        Path to extracted PetFinder data directory.

    Raises:
        RuntimeError: If Kaggle CLI fails or is not installed.
    """
    target = data_dir / "petfinder"
    if (target / "train.csv").exists():
        logger.info("PetFinder data already exists at %s, skipping", target)
        return target

    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading PetFinder dataset via Kaggle API...")

    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                PETFINDER_COMPETITION,
                "-p",
                str(target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as err:
        raise RuntimeError(
            "Kaggle CLI not found. Install with: pip install kaggle"
        ) from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Kaggle download failed: {err.stderr}"
        ) from err

    zip_path = target / f"{PETFINDER_COMPETITION}.zip"
    if zip_path.exists():
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


def _download_file(url: str, dest: Path) -> None:
    """Download a file with progress bar.

    Args:
        url: URL to download from.
        dest: Destination file path.

    Raises:
        RuntimeError: If download fails.
    """
    logger.info("Downloading %s...", url.split("/")[-1])
    try:
        response = requests.get(url, stream=True, timeout=300)
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
