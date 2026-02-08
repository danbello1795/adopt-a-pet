"""Tests for src/config.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import Config, get_config


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = Config()
        assert config.index_name == "pets"
        assert config.clip_model_name == "ViT-B-32"
        assert config.clip_pretrained == "laion2b_s34b_b79k"
        assert config.embedding_dim == 512
        assert config.petfinder_sample_size == 1000
        assert config.oxford_sample_size == 500
        assert config.random_seed == 42
        assert config.default_top_k == 20
        assert config.knn_num_candidates == 100

    def test_data_dir_default(self) -> None:
        """Data dir should default to 'data'."""
        config = Config()
        assert config.data_dir == Path("data")

    def test_data_dir_from_env(self, monkeypatch: object) -> None:
        """Data dir should read from DATA_DIR env var."""
        monkeypatch.setattr(
            os, "getenv", lambda k, d=None: "/custom/data" if k == "DATA_DIR" else d
        )
        config = Config()
        assert config.data_dir == Path("/custom/data")

    def test_elasticsearch_url_default(self) -> None:
        """ES URL should default to localhost:9200."""
        config = Config()
        assert config.elasticsearch_url == "http://localhost:9200"

    def test_frozen_dataclass(self) -> None:
        """Config should be immutable (frozen)."""
        config = Config()
        with pytest.raises(AttributeError):
            config.index_name = "other"  # type: ignore[misc]

    def test_get_config_returns_config(self) -> None:
        """get_config should return a Config instance."""
        config = get_config()
        assert isinstance(config, Config)
