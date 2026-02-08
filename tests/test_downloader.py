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

    @patch("src.data.downloader.subprocess.run")
    def test_calls_kaggle_cli(self, mock_run: MagicMock, tmp_data_dir: Path) -> None:
        """Should invoke kaggle CLI with correct arguments."""
        mock_run.return_value = MagicMock(returncode=0)

        try:
            download_petfinder(tmp_data_dir)
        except Exception:
            pass

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "kaggle" in call_args
        assert "petfinder-adoption-prediction" in call_args

    @patch("src.data.downloader.subprocess.run")
    def test_raises_on_missing_kaggle(
        self, mock_run: MagicMock, tmp_data_dir: Path
    ) -> None:
        """Should raise RuntimeError if kaggle CLI not found."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(RuntimeError, match="Kaggle CLI not found"):
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
