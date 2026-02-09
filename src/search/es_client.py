"""Elasticsearch client wrapper with health checks."""

from __future__ import annotations

import logging
import time

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


def _build_es_client(
    url: str = "http://localhost:9200",
    cloud_id: str | None = None,
    api_key: str | None = None,
) -> Elasticsearch:
    """Build an Elasticsearch client for local or Elastic Cloud.

    Args:
        url: Elasticsearch URL (ignored when cloud_id is set).
        cloud_id: Elastic Cloud deployment ID.
        api_key: Elastic Cloud API key.

    Returns:
        Elasticsearch client instance (not yet verified).
    """
    if cloud_id:
        return Elasticsearch(cloud_id=cloud_id, api_key=api_key)
    return Elasticsearch(url)


def create_es_client(
    url: str = "http://localhost:9200",
    cloud_id: str | None = None,
    api_key: str | None = None,
) -> Elasticsearch:
    """Create and verify an Elasticsearch client connection.

    Args:
        url: Elasticsearch URL (ignored when cloud_id is set).
        cloud_id: Elastic Cloud deployment ID.
        api_key: Elastic Cloud API key.

    Returns:
        Connected Elasticsearch client.

    Raises:
        ConnectionError: If unable to connect to Elasticsearch.
    """
    es = _build_es_client(url, cloud_id, api_key)
    target = cloud_id or url
    if not es.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {target}")
    logger.info("Connected to Elasticsearch at %s", target)
    return es


def wait_for_elasticsearch(
    url: str,
    timeout: int = 120,
    cloud_id: str | None = None,
    api_key: str | None = None,
) -> bool:
    """Wait for Elasticsearch to become healthy.

    Retries connection every 5 seconds until timeout.

    Args:
        url: Elasticsearch URL (ignored when cloud_id is set).
        timeout: Maximum seconds to wait.
        cloud_id: Elastic Cloud deployment ID.
        api_key: Elastic Cloud API key.

    Returns:
        True if ES is healthy, False if timeout reached.
    """
    es = _build_es_client(url, cloud_id, api_key)
    target = cloud_id or url
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        try:
            if es.ping():
                logger.info("Elasticsearch is ready at %s", target)
                return True
        except Exception:
            pass
        logger.info("Waiting for Elasticsearch...")
        time.sleep(5)

    logger.error("Elasticsearch not available at %s after %ds", target, timeout)
    return False
