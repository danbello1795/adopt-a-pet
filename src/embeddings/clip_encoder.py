"""CLIP model wrapper for text and image encoding."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """Encode text and images into CLIP embedding space.

    Uses ViT-B-32 model producing 512-dimensional L2-normalized vectors.
    Both text and image embeddings live in the same vector space,
    enabling cross-modal similarity search.

    Args:
        model_name: CLIP model architecture name.
        pretrained: Pretrained weights identifier.
        device: Device to run model on (auto-detected if None).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading CLIP model %s (%s) on %s",
            model_name,
            pretrained,
            self.device,
        )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embedding_dim = 512

        logger.info("CLIP model loaded successfully")

    def encode_text(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode text descriptions into normalized 512-dim vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Number of texts to process at once.

        Returns:
            List of 512-dimensional float vectors.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(batch).to(self.device)

            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.extend(features.cpu().numpy().tolist())

        return all_embeddings

    def encode_images(
        self,
        image_paths: list[str | Path],
        batch_size: int = 16,
    ) -> list[list[float]]:
        """Encode images into normalized 512-dim vectors.

        Args:
            image_paths: List of paths to image files.
            batch_size: Number of images to process at once.

        Returns:
            List of 512-dimensional float vectors.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception:
                    logger.warning("Failed to load image: %s", path)
                    images.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.extend(features.cpu().numpy().tolist())

        return all_embeddings

    def encode_single_image(self, image: Image.Image) -> list[float]:
        """Encode a single PIL Image for upload-based search.

        Args:
            image: PIL Image object (e.g., from user upload).

        Returns:
            512-dimensional float vector.
        """
        img_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().tolist()[0]
