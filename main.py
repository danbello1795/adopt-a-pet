#!/usr/bin/env python3
"""Adopt-a-Pet: Single entry point.

Downloads data, generates CLIP embeddings, indexes into Elasticsearch,
and launches the FastAPI web UI.

Usage:
    python main.py
    python main.py --skip-download
    python main.py --skip-index
    python main.py --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("adopt-a-pet")


def main() -> None:
    """Orchestrate the full pipeline: download -> process -> embed -> index -> serve."""
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
    args = parser.parse_args()

    from src.config import get_config

    config = get_config()
    es_url = args.es_url or config.elasticsearch_url

    # Step 1: Wait for Elasticsearch
    logger.info("Step 1/5: Connecting to Elasticsearch at %s", es_url)
    from src.search.es_client import wait_for_elasticsearch

    if not wait_for_elasticsearch(es_url):
        logger.error("Elasticsearch not available. Exiting.")
        sys.exit(1)

    # Step 2: Download data
    if not args.skip_download:
        logger.info("Step 2/5: Downloading datasets...")
        from src.data.downloader import download_oxford_pets, download_petfinder

        config.data_dir.mkdir(parents=True, exist_ok=True)
        petfinder_dir = download_petfinder(config.data_dir)
        oxford_dir = download_oxford_pets(config.data_dir)
    else:
        logger.info("Step 2/5: Skipping download (--skip-download)")
        petfinder_dir = config.data_dir / "petfinder"
        oxford_dir = config.data_dir / "oxford_pets"

    # Step 3: Process and merge data
    if not args.skip_index:
        logger.info("Step 3/5: Processing datasets...")
        from src.data.processor import (
            merge_datasets,
            process_oxford,
            process_petfinder,
        )

        pf_records = process_petfinder(
            petfinder_dir, sample_size=config.petfinder_sample_size
        )
        ox_records = process_oxford(
            oxford_dir, sample_size=config.oxford_sample_size
        )
        all_records = merge_datasets(pf_records, ox_records)
        logger.info("Total records: %d", len(all_records))

        # Step 4: Generate CLIP embeddings
        logger.info("Step 4/5: Generating CLIP embeddings...")
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
        logger.info("Step 5/5: Indexing into Elasticsearch...")
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
    logger.info("Launching web UI on %s:%d", args.host, args.port)
    import uvicorn

    from src.api.app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
