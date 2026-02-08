"""Elasticsearch client wrapper with health checks."""

from __future__ import annotations

import logging
import time

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


def create_es_client(url: str = "http://localhost:9200") -> Elasticsearch:
    """Create and verify an Elasticsearch client connection.

    Args:
        url: Elasticsearch URL.

    Returns:
        Connected Elasticsearch client.

    Raises:
        ConnectionError: If unable to connect to Elasticsearch.
    """
    es = Elasticsearch(url)
    if not es.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {url}")
    logger.info("Connected to Elasticsearch at %s", url)
    return es


def wait_for_elasticsearch(url: str, timeout: int = 120) -> bool:
    """Wait for Elasticsearch to become healthy.

    Retries connection every 5 seconds until timeout.

    Args:
        url: Elasticsearch URL.
        timeout: Maximum seconds to wait.

    Returns:
        True if ES is healthy, False if timeout reached.
    """
    es = Elasticsearch(url)
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        try:
            if es.ping():
                logger.info("Elasticsearch is ready at %s", url)
                return True
        except Exception:
            pass
        logger.info("Waiting for Elasticsearch...")
        time.sleep(5)

    logger.error("Elasticsearch not available at %s after %ds", url, timeout)
    return False
