"""Tests for src/embeddings/clip_encoder.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def mock_clip_encoder():
    """Create a CLIPEncoder with mocked open_clip module."""
    mock_oc = MagicMock()
    mock_model = MagicMock()
    mock_preprocess = MagicMock()
    mock_tokenizer = MagicMock()

    fake_features = torch.randn(1, 512)
    fake_features = fake_features / fake_features.norm(dim=-1, keepdim=True)
    mock_model.encode_text.return_value = fake_features
    mock_model.encode_image.return_value = fake_features
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    mock_preprocess.return_value = torch.randn(3, 224, 224)

    mock_oc.create_model_and_transforms.return_value = (
        mock_model,
        None,
        mock_preprocess,
    )
    mock_oc.get_tokenizer.return_value = mock_tokenizer
    mock_tokenizer.return_value = torch.zeros(1, 77, dtype=torch.long)

    with patch.dict(sys.modules, {"open_clip": mock_oc}):
        from src.embeddings.clip_encoder import CLIPEncoder

        encoder = CLIPEncoder(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device="cpu",
        )

    return encoder


class TestCLIPEncoder:
    """Tests for CLIPEncoder class."""

    def test_init_sets_device(self, mock_clip_encoder) -> None:
        """Should set device to cpu when specified."""
        assert mock_clip_encoder.device == "cpu"

    def test_embedding_dim(self, mock_clip_encoder) -> None:
        """Should have 512-dim embeddings."""
        assert mock_clip_encoder.embedding_dim == 512

    def test_encode_text_returns_list(self, mock_clip_encoder) -> None:
        """Should return list of float vectors."""
        result = mock_clip_encoder.encode_text(["a friendly dog"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 512

    def test_encode_text_batch(self, mock_clip_encoder) -> None:
        """Should handle batched text encoding."""
        texts = ["dog", "cat", "bird"]
        batch_features = torch.randn(3, 512)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        mock_clip_encoder.model.encode_text.return_value = batch_features
        mock_clip_encoder.tokenizer.return_value = torch.zeros(3, 77, dtype=torch.long)

        result = mock_clip_encoder.encode_text(texts)
        assert len(result) == 3

    def test_encode_text_normalized(self, mock_clip_encoder) -> None:
        """Embeddings should be approximately L2-normalized."""
        result = mock_clip_encoder.encode_text(["test query"])
        vector = np.array(result[0])
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 0.01

    def test_encode_single_image(self, mock_clip_encoder) -> None:
        """Should encode a single PIL image."""
        image = Image.new("RGB", (224, 224), color="red")
        result = mock_clip_encoder.encode_single_image(image)
        assert isinstance(result, list)
        assert len(result) == 512

    def test_encode_single_image_normalized(self, mock_clip_encoder) -> None:
        """Single image embedding should be L2-normalized."""
        image = Image.new("RGB", (224, 224), color="blue")
        result = mock_clip_encoder.encode_single_image(image)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01

    def test_encode_images_batch(self, mock_clip_encoder, tmp_path: Path) -> None:
        """Should encode a batch of images from file paths."""
        paths = []
        for i in range(3):
            img = Image.new("RGB", (100, 100), color="green")
            p = tmp_path / f"img_{i}.jpg"
            img.save(p)
            paths.append(str(p))

        batch_features = torch.randn(3, 512)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        mock_clip_encoder.model.encode_image.return_value = batch_features

        result = mock_clip_encoder.encode_images(paths)
        assert len(result) == 3
        assert all(len(v) == 512 for v in result)
