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
- Docker and Docker Compose
- Kaggle API credentials ([setup guide](https://www.kaggle.com/settings))

### Option 1: Docker Compose (recommended)

```bash
# Configure environment
cp .env.example .env
# Edit .env with your Kaggle credentials

# Launch
docker compose up

# Open http://localhost:8000
```

### Option 2: Local Development

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv automatically)
uv sync

# Start Elasticsearch
docker compose up elasticsearch -d

# Run the application
uv run python main.py
```

### CLI Options

```bash
python main.py                    # Full pipeline: download, embed, index, serve
python main.py --skip-download    # Skip data download (use cached data)
python main.py --skip-index       # Skip embedding and indexing
python main.py --port 9000        # Custom port
python main.py --es-url http://...  # Custom Elasticsearch URL
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
- **Search**: Elasticsearch 8.12.0 (kNN dense_vector, cosine similarity)
- **Web**: FastAPI, Jinja2, Tailwind CSS
- **Data**: pandas, Pydantic, Kaggle API
- **Testing**: pytest, httpx, 85 tests with mocked dependencies

For detailed technical decisions and design rationale, see [docs/technical_design.md](docs/technical_design.md).
