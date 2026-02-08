# Technical Design Document: Adopt-a-Pet Cross-Modal Search

## 1. Overview

Adopt-a-Pet is a cross-modal search system that enables users to find adoptable pets through natural language text queries or image uploads. The system leverages CLIP (Contrastive Language-Image Pretraining) embeddings for unified text-image representation and Elasticsearch kNN for fast vector similarity search.

## 2. Architecture

```
User (Browser)
    |
    v
FastAPI + Jinja2 (port 8000)
    |
    v
PetSearcher
    |
    +-- CLIPEncoder (ViT-B-32, 512-dim)
    |     - encode_text(): text -> 512-dim vector
    |     - encode_single_image(): PIL image -> 512-dim vector
    |
    +-- Elasticsearch 8.x (kNN dense_vector)
          - text_embedding (512-dim, cosine)
          - image_embedding (512-dim, cosine)
```

### Search Flows

**Text Search:**
1. User submits text query (e.g., "playful ginger cat")
2. CLIPEncoder.encode_text() produces a 512-dim vector
3. Elasticsearch kNN query searches both text_embedding (boost=1.5) and image_embedding (boost=1.0)
4. Results ranked by combined cosine similarity score

**Image Search:**
1. User uploads a pet photo
2. CLIPEncoder.encode_single_image() produces a 512-dim vector
3. Elasticsearch kNN query searches image_embedding (boost=2.0) and text_embedding (boost=0.5)
4. Results ranked by combined cosine similarity score

## 3. Data Pipeline

### Data Sources

| Source | Records | Content |
|--------|---------|---------|
| PetFinder.my (Kaggle) | ~1000 sampled | Descriptions, breed, age, gender, photos |
| Oxford-IIIT Pet Dataset | ~500 sampled | Breed-labeled pet photos with synthetic descriptions |

### Processing Pipeline

1. **Download**: PetFinder via Kaggle API, Oxford-IIIT via HTTP
2. **Process**: Join breed/color labels, filter records with descriptions and photos, sample
3. **Normalize**: Map both datasets to unified `PetRecord` schema
4. **Embed**: Generate CLIP text + image embeddings for each record
5. **Index**: Bulk index into Elasticsearch with dual embedding fields

### Unified Schema: PetRecord

| Field | Type | Description |
|-------|------|-------------|
| pet_id | str | Unique ID prefixed by source (pf-, ox-) |
| source | str | "petfinder" or "oxford_iiit" |
| name | str | Pet name or breed name |
| species | str | "Dog" or "Cat" |
| breed | str | Primary breed name |
| age_months | int? | Age in months (PetFinder only) |
| gender | str? | Male, Female, or None |
| description | str | Text for CLIP encoding |
| image_path | str | Path to primary image |
| metadata | dict | Source-specific fields |

## 4. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CLIP model | ViT-B-32 via open-clip-torch | Fast inference, 512-dim vectors, actively maintained |
| Elasticsearch version | 8.12.0 | Native kNN dense_vector support |
| Similarity metric | Cosine | CLIP vectors are L2-normalized |
| Multi-kNN strategy | Dual fields with boosting | Different weight per query type |
| UI framework | FastAPI + Jinja2 + Tailwind CSS | No build step, fast to develop |
| PyTorch in Docker | CPU-only | ~2GB image vs ~5GB with CUDA |
| Data sampling | 1000 PF + 500 Oxford | Lightweight but representative PoC |

## 5. Elasticsearch Index Design

```json
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "pet_id": { "type": "keyword" },
      "source": { "type": "keyword" },
      "name": { "type": "text" },
      "species": { "type": "keyword" },
      "breed": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "text_embedding": { "type": "dense_vector", "dims": 512, "index": true, "similarity": "cosine" },
      "image_embedding": { "type": "dense_vector", "dims": 512, "index": true, "similarity": "cosine" }
    }
  }
}
```

### kNN Query Strategy

Text queries use higher boost on text_embedding (1.5 vs 1.0) because CLIP text-to-text alignment is stronger than text-to-image for descriptive queries. Image queries reverse this (image_embedding boost=2.0, text=0.5) because visual similarity is the primary signal.

Elasticsearch combines multiple kNN clauses via linear score combination, producing a single ranked result set.

## 6. CLIP Encoding Details

- **Model**: ViT-B-32 pretrained on LAION-2B
- **Text encoding**: Tokenizes to 77 tokens max, produces 512-dim L2-normalized vector
- **Image encoding**: Resizes/crops to 224x224, produces 512-dim L2-normalized vector
- **Batch processing**: Text (batch=32), Images (batch=16) to manage memory
- **Text optimization**: Descriptions combine structured metadata (breed, species, age) with truncated free text to stay within 77-token limit

## 7. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Home page with search UI |
| GET | /search?q=&top_k=20 | Text search with HTML results |
| POST | /search/image | Image upload search with HTML results |
| GET | /api/search?q=&top_k=20 | JSON API for text search |
| GET | /health | Health check (ES connectivity) |

## 8. Deployment

### Docker Compose Services

- **elasticsearch**: ES 8.12.0, single-node, security disabled, 512MB heap
- **app**: Python 3.12-slim, CPU-only PyTorch, depends on ES health

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| KAGGLE_KEY | - | Kaggle API key (pre-configured in .env) |
| ELASTICSEARCH_URL | http://localhost:9200 | ES connection URL |
| DATA_DIR | data | Data storage directory |
| HOST | 0.0.0.0 | Server bind host |
| PORT | 8000 | Server bind port |

## 9. Testing Strategy

- **Unit tests**: Schemas, config, processor, encoder (mocked), indexer (mocked), searcher (mocked)
- **Integration tests**: FastAPI routes with TestClient and mocked search backend
- **Coverage target**: >80% via pytest-cov
- **Mocking approach**: Elasticsearch client, CLIP model, and external downloads are mocked to enable fast, offline testing

## 10. Known Limitations

- CLIP truncates text to 77 tokens; long descriptions are truncated
- PetFinder images use only the first photo per listing
- Oxford-IIIT descriptions are synthetic (breed + species template)
- No GPU acceleration in Docker (CPU-only PyTorch)
- Single ES node with no replicas (PoC, not production)
- No authentication or rate limiting on API endpoints
