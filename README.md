# MLE Test: Adopt a Pet

Cross-modal pet search system using CLIP embeddings and Elasticsearch kNN. Search for adoptable pets by text description or by uploading a photo.

## Architecture

```
Browser --> FastAPI + Jinja2 UI --> PetSearcher
                                      |
                                      +-- CLIPEncoder (ViT-B-32, 512-dim)
                                      +-- Elasticsearch 8.x (kNN dense_vector)
```

- **Text search**: query -> CLIP text encoder -> kNN on text_embedding (boost=1.5) + image_embedding (boost=1.0)
- **Image search**: upload -> CLIP image encoder -> kNN on image_embedding (boost=2.0) + text_embedding (boost=0.5)

## Project Structure

```
adopt-a-pet/
├── src/
│   ├── config.py                  # Central configuration
│   ├── data/
│   │   ├── downloader.py          # Dataset download (Kaggle + HTTP)
│   │   ├── processor.py           # Process, sample, merge datasets
│   │   └── schemas.py             # Pydantic models
│   ├── embeddings/
│   │   └── clip_encoder.py        # CLIP text/image encoder
│   ├── search/
│   │   ├── es_client.py           # Elasticsearch connection
│   │   ├── indexer.py             # Index creation, bulk indexing
│   │   └── searcher.py            # kNN search engine
│   └── api/
│       ├── app.py                 # FastAPI app factory
│       ├── routes.py              # API routes
│       └── templates/             # Jinja2 HTML templates
├── tests/                         # Test suite (85 tests)
├── docs/
│   └── technical_design.md        # Full technical design document
├── docker/
│   └── Dockerfile                 # Multi-stage Python 3.12 build
├── docker-compose.yml             # ES + App services
├── main.py                        # Single entry point
├── pyproject.toml                 # Dependencies and tool config
└── .env.example                   # Environment variable template
```

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (for Elasticsearch)
- Kaggle API credentials (optional, for PetFinder dataset — [setup guide](https://www.kaggle.com/settings))

### One-command launch

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv automatically)
uv sync

# Launch everything (auto-starts Elasticsearch via Docker)
uv run python main.py
```

`main.py` detects whether Elasticsearch is running. If it isn't, it starts it
automatically via `docker compose up -d elasticsearch`, waits for it to become
healthy, downloads the datasets, generates CLIP embeddings, indexes them, and
launches the web UI at **http://localhost:8000**.

> **Note:** Kaggle credentials are only needed for PetFinder data. Without them,
> the pipeline continues with the Oxford-IIIT dataset alone (~500 records).
> To enable PetFinder, the `KAGGLE_KEY` is already configured in `.env`.

### Full Docker Compose (app + ES in containers)

```bash
# Configure environment
cp .env.example .env
# Edit .env with your Kaggle credentials

# Launch both services
docker compose up

# Open http://localhost:8000
```

### Pipeline steps

`main.py` runs six steps in order:

| Step | Description | Skip flag |
|------|-------------|-----------|
| 1 | Ensure Elasticsearch is running (auto-start via Docker if needed) | `--no-docker` |
| 2 | Download PetFinder (Kaggle) and Oxford-IIIT datasets | `--skip-download` |
| 3 | Process and merge datasets into `PetRecord` objects | `--skip-index` |
| 4 | Generate CLIP embeddings (text + image) | `--skip-index` |
| 5 | Bulk-index documents into Elasticsearch | `--skip-index` |
| 6 | Launch FastAPI web UI | — |

### CLI Options

```bash
uv run python main.py                       # Full pipeline
uv run python main.py --skip-download       # Use already-downloaded data
uv run python main.py --skip-index          # Skip embeddings + indexing (data already indexed)
uv run python main.py --no-docker           # Don't auto-start ES (manage it yourself)
uv run python main.py --port 9000           # Custom server port
uv run python main.py --host 127.0.0.1      # Custom server host
uv run python main.py --es-url http://...   # Custom Elasticsearch URL
```

## Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| [PetFinder.my](https://www.kaggle.com/c/petfinder-adoption-prediction) | ~1000 | Real adoption listings with descriptions and photos |
| [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) | ~500 | Breed-labeled pet photos with synthetic descriptions |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Home page with search UI |
| GET | `/search?q=friendly+cat` | Text search (HTML) |
| POST | `/search/image` | Image upload search (HTML) |
| GET | `/api/search?q=cat&top_k=10` | Text search (JSON) |
| GET | `/health` | Health check |

## Development

### Running Tests

```bash
uv run pytest tests/ -v --cov=src
```

### Code Quality

```bash
uv run black src/ tests/
uv run isort src/ tests/
uv run ruff check src/ tests/
```

### Tech Stack

- **ML**: open-clip-torch (ViT-B-32), PyTorch, Pillow
- **Search**: Elasticsearch 8.12.0 (kNN dense_vector, cosine similarity), elasticsearch-py 8.x
- **Web**: FastAPI, Jinja2, Tailwind CSS
- **Data**: pandas, Pydantic, Kaggle API
- **Testing**: pytest, httpx, 85 tests with mocked dependencies

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `Elasticsearch not available` | Ensure Docker is running. If port 9200 is occupied by another container, stop it with `docker stop <name>` |
| `Kaggle download failed` | Verify `KAGGLE_KEY` in `.env` is valid. Without it, only Oxford-IIIT data is used |
| Port 8000 already in use | Use `--port 9000` or stop the process on 8000 |
| ES client returns 400 | Ensure `elasticsearch` Python package is `<9` (pinned in pyproject.toml). Client 9.x is not compatible with ES 8.x |

For detailed technical decisions and design rationale, see [docs/technical_design.md](docs/technical_design.md).
