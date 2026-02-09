"""Tests for src/data/downloader.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.downloader import download_oxford_pets, download_petfinder


class TestDownloadPetfinder:
    """Tests for PetFinder download functionality."""

    def test_skip_existing_data(self, tmp_data_dir: Path) -> None:
        """Should skip download if train.csv already exists."""
        pf_dir = tmp_data_dir / "petfinder"
        pf_dir.mkdir()
        (pf_dir / "train.csv").write_text("header\n")

        result = download_petfinder(tmp_data_dir)
        assert result == pf_dir

    @patch("pathlib.Path.unlink")
    @patch("src.data.downloader._extract_zip")
    @patch("src.data.downloader._download_file")
    @patch("src.data.downloader._get_kaggle_key", return_value="test-key")
    def test_calls_kaggle_api(
        self,
        mock_key: MagicMock,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_unlink: MagicMock,
        tmp_data_dir: Path,
    ) -> None:
        """Should call Kaggle REST API with bearer token."""
        download_petfinder(tmp_data_dir)

        mock_key.assert_called_once()
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args
        assert "petfinder-adoption-prediction" in str(call_kwargs)
        assert call_kwargs[1]["bearer_token"] == "test-key"

    @patch("src.data.downloader._get_kaggle_key")
    def test_raises_on_missing_credentials(
        self, mock_key: MagicMock, tmp_data_dir: Path
    ) -> None:
        """Should raise RuntimeError if Kaggle credentials not found."""
        mock_key.side_effect = RuntimeError("Kaggle credentials not found")

        with pytest.raises(RuntimeError, match="Kaggle credentials not found"):
            download_petfinder(tmp_data_dir)


class TestDownloadOxford:
    """Tests for Oxford-IIIT download functionality."""

    def test_skip_existing_data(self, tmp_data_dir: Path) -> None:
        """Should skip download if images and annotations exist."""
        ox_dir = tmp_data_dir / "oxford_pets"
        (ox_dir / "images").mkdir(parents=True)
        (ox_dir / "annotations").mkdir(parents=True)

        result = download_oxford_pets(tmp_data_dir)
        assert result == ox_dir

    @patch("pathlib.Path.unlink")
    @patch("src.data.downloader._extract_tar")
    @patch("src.data.downloader._download_file")
    def test_downloads_two_archives(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_unlink: MagicMock,
        tmp_data_dir: Path,
    ) -> None:
        """Should download images and annotations archives."""
        download_oxford_pets(tmp_data_dir)
        assert mock_download.call_count == 2
        assert mock_extract.call_count == 2
