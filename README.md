# MLE Test: Adopt a Pet

Pet adoption recommendation system - ML Engineering test project.

## Project Overview

This project aims to build a machine learning system that recommends pets for adoption based on user preferences and pet characteristics.

## Project Structure

```
adopt-a-pet/
├── src/               # Source code
├── tests/             # Test files
├── docs/              # Documentation
├── scripts/           # Utility scripts
├── notebooks/         # Jupyter notebooks
├── docker/            # Docker-related files
├── .github/           # GitHub workflows
├── pyproject.toml     # Project configuration & dependencies
├── .gitignore         # Git ignore patterns
└── README.md          # This file
```

## Setup

### Prerequisites

- Python 3.11+
- uv package manager

### Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install uv (if not already installed):
```bash
pip install uv
```

3. Install dependencies:
```bash
uv pip install -e ".[dev]"
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

### Linting

```bash
ruff check src/ tests/
mypy src/
```

## Git Workflow

- Create feature branches: `git checkout -b feature/descriptive-name`
- Branch naming: `feature/`, `bugfix/`, `refactor/`, `docs/`
- Commit format: `<type>: <description>` (feat, fix, refactor, docs, test, chore)
- Never commit directly to `main`

## Quality Checks

Before committing, ensure:
1. Dependencies are synced: `uv pip sync pyproject.toml`
2. Tests pass: `pytest tests/`
3. Code is formatted: `black src/ tests/ && isort src/ tests/`
4. No linting errors: `ruff check src/ tests/`
5. Type checking passes: `mypy src/`

## License

[Add your license here]
