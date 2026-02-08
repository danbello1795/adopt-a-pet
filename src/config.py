"""Central application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Central application configuration.

    Reads from environment variables with sensible defaults.
    All paths are resolved relative to project root.
    """

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))

    # Elasticsearch
    elasticsearch_url: str = field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    )
    index_name: str = "pets"

    # CLIP model
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    embedding_dim: int = 512

    # Data sampling
    petfinder_sample_size: int = 1000
    oxford_sample_size: int = 500
    random_seed: int = 42

    # Search
    default_top_k: int = 20
    knn_num_candidates: int = 100

    # Server
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))


def get_config() -> Config:
    """Get application configuration.

    Returns:
        Config instance with values from environment or defaults.
    """
    return Config()
