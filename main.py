#!/usr/bin/env python3
"""Adopt-a-Pet: Single entry point.

Downloads data, generates CLIP embeddings, indexes into Elasticsearch,
and launches the FastAPI web UI.

Automatically starts Elasticsearch via docker-compose if it is not already
running, so a bare ``python main.py`` is enough to launch the full stack.

Usage:
    python main.py
    python main.py --skip-download
    python main.py --skip-index
    python main.py --port 8000
    python main.py --no-docker          # don't auto-start ES
"""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("adopt-a-pet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _es_is_reachable(url: str) -> bool:
    """Return True if Elasticsearch responds to a ping at *url*."""
    from elasticsearch import Elasticsearch

    try:
        return Elasticsearch(url).ping()
    except Exception:
        return False


def _is_docker_daemon_running() -> bool:
    """Return True if the Docker daemon is responding."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _start_docker_desktop(timeout: int = 60) -> bool:
    """Try to launch Docker Desktop and wait until the daemon is ready.

    Args:
        timeout: Max seconds to wait for the daemon after launching.

    Returns:
        True if the daemon became reachable within *timeout*.
    """
    if platform.system() != "Windows":
        logger.warning("Auto-start Docker Desktop is only supported on Windows.")
        return False

    docker_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if not shutil.which("docker") and not __import__("pathlib").Path(docker_path).exists():
        return False

    logger.info("Docker daemon not running. Starting Docker Desktop...")
    try:
        subprocess.Popen(
            [docker_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        logger.warning("Could not launch Docker Desktop: %s", exc)
        return False

    logger.info("Waiting up to %ds for Docker daemon to be ready...", timeout)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_docker_daemon_running():
            logger.info("Docker daemon is ready.")
            return True
        time.sleep(3)

    logger.error("Docker daemon did not start within %ds.", timeout)
    return False


def _start_elasticsearch_docker() -> None:
    """Start only the ``elasticsearch`` service via docker-compose.

    If the Docker daemon is not running, attempts to launch Docker Desktop
    first (Windows only) before retrying.
    """
    compose_cmd = _find_compose_command()
    if compose_cmd is None:
        logger.error(
            "docker-compose / docker compose not found. "
            "Please start Elasticsearch manually or install Docker."
        )
        sys.exit(1)

    logger.info("Starting Elasticsearch via %s ...", " ".join(compose_cmd))
    try:
        subprocess.run(
            [*compose_cmd, "up", "-d", "elasticsearch"],
            check=True,
        )
    except subprocess.CalledProcessError:
        if not _is_docker_daemon_running():
            if not _start_docker_desktop():
                logger.error(
                    "Could not start Docker Desktop. "
                    "Please start it manually and retry."
                )
                sys.exit(1)
            # Retry after Docker Desktop is ready
            logger.info("Retrying: starting Elasticsearch via %s ...", " ".join(compose_cmd))
            try:
                subprocess.run(
                    [*compose_cmd, "up", "-d", "elasticsearch"],
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.error("Failed to start Elasticsearch (exit code %d)", exc.returncode)
                sys.exit(1)
        else:
            logger.error("Docker is running but failed to start Elasticsearch.")
            sys.exit(1)


def _find_compose_command() -> list[str] | None:
    """Return the docker-compose CLI invocation available on this system."""
    if shutil.which("docker"):
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return ["docker", "compose"]

    if shutil.which("docker-compose"):
        return ["docker-compose"]

    return None


def _open_browser(url: str, delay: float = 2.0) -> None:
    """Open browser after a delay to give the server time to start.

    Args:
        url: URL to open in the browser.
        delay: Seconds to wait before opening.
    """
    def _delayed_open():
        time.sleep(delay)
        logger.info("Opening browser at %s", url)
        webbrowser.open(url)

    thread = threading.Thread(target=_delayed_open, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full pipeline: docker -> download -> process -> embed -> index -> serve."""
    parser = argparse.ArgumentParser(
        description="Adopt-a-Pet Cross-Modal Search System"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download (use cached data)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip embedding generation and ES indexing",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host"
    )
    parser.add_argument(
        "--es-url", type=str, default=None, help="Elasticsearch URL"
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Do not auto-start Elasticsearch via docker-compose",
    )
    args = parser.parse_args()

    from src.config import get_config

    config = get_config()
    es_url = args.es_url or config.elasticsearch_url

    # Step 1: Ensure Elasticsearch is running
    logger.info("Step 1/6: Ensuring Elasticsearch is available at %s", es_url)

    if not _es_is_reachable(es_url):
        if args.no_docker:
            logger.error(
                "Elasticsearch is not reachable at %s and --no-docker was set. "
                "Start Elasticsearch manually and retry.",
                es_url,
            )
            sys.exit(1)

        logger.info("Elasticsearch not reachable – starting via docker-compose...")
        _start_elasticsearch_docker()

    from src.search.es_client import wait_for_elasticsearch

    if not wait_for_elasticsearch(es_url):
        logger.error("Elasticsearch not available after waiting. Exiting.")
        sys.exit(1)

    # Step 2: Download data
    petfinder_dir = config.data_dir / "petfinder"
    oxford_dir = config.data_dir / "oxford_pets"

    if not args.skip_download:
        logger.info("Step 2/6: Downloading datasets...")
        from src.data.downloader import download_oxford_pets, download_petfinder

        config.data_dir.mkdir(parents=True, exist_ok=True)

        try:
            petfinder_dir = download_petfinder(config.data_dir)
        except Exception as exc:
            logger.warning("PetFinder download failed: %s", exc)
            logger.warning(
                "Continuing without PetFinder data. "
                "Set KAGGLE_KEY in .env or configure ~/.kaggle/kaggle.json."
            )

        try:
            oxford_dir = download_oxford_pets(config.data_dir)
        except Exception as exc:
            logger.warning("Oxford-IIIT download failed: %s", exc)
    else:
        logger.info("Step 2/6: Skipping download (--skip-download)")

    # Step 3: Process and merge data
    if not args.skip_index:
        logger.info("Step 3/6: Processing datasets...")
        from src.data.processor import (
            merge_datasets,
            process_oxford,
            process_petfinder,
        )

        pf_records: list = []
        ox_records: list = []

        petfinder_has_data = (petfinder_dir / "train.csv").exists() or (
            petfinder_dir / "train" / "train.csv"
        ).exists()
        if petfinder_dir.exists() and petfinder_has_data:
            pf_records = process_petfinder(
                petfinder_dir, sample_size=config.petfinder_sample_size
            )
        else:
            logger.warning("PetFinder data not found at %s, skipping", petfinder_dir)

        if oxford_dir.exists() and (oxford_dir / "annotations").exists():
            ox_records = process_oxford(
                oxford_dir, sample_size=config.oxford_sample_size
            )
        else:
            logger.warning("Oxford-IIIT data not found at %s, skipping", oxford_dir)

        all_records = merge_datasets(pf_records, ox_records)

        if not all_records:
            logger.error(
                "No data available to index. Download at least one dataset first."
            )
            sys.exit(1)
        logger.info("Total records: %d", len(all_records))

        # Step 4: Generate CLIP embeddings
        logger.info("Step 4/6: Generating CLIP embeddings...")
        from src.embeddings.clip_encoder import CLIPEncoder

        encoder = CLIPEncoder(
            model_name=config.clip_model_name,
            pretrained=config.clip_pretrained,
        )

        descriptions = [r.description for r in all_records]
        image_paths = [r.image_path for r in all_records]

        logger.info("Encoding %d text descriptions...", len(descriptions))
        text_embeddings = encoder.encode_text(descriptions)

        logger.info("Encoding %d images...", len(image_paths))
        image_embeddings = encoder.encode_images(image_paths)

        # Step 5: Index into Elasticsearch
        logger.info("Step 5/6: Indexing into Elasticsearch...")
        from elasticsearch import Elasticsearch

        from src.search.indexer import create_index, index_pets

        es = Elasticsearch(es_url)
        create_index(es, config.index_name)
        indexed = index_pets(
            es, all_records, text_embeddings, image_embeddings,
            index_name=config.index_name,
        )
        logger.info("Indexed %d documents", indexed)
        es.close()
    else:
        logger.info("Steps 3-5: Skipping indexing (--skip-index)")

    # Step 6: Launch FastAPI server
    logger.info("Step 6/6: Launching web UI on %s:%d", args.host, args.port)
    import uvicorn

    from src.api.app import create_app

    app = create_app()

    # Open browser automatically
    url = f"http://localhost:{args.port}"
    _open_browser(url)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
