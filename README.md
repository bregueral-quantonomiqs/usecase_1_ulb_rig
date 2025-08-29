# Insurance Analysis Project

A Python project for analyzing insurance claim data.

## Project Structure

```
.
├── data/               # Data files
│   └── mcc.csv         # Insurance claim data
├── docs/               # Documentation
├── src/                # Source code
│   └── insurance_analysis/  # Main package
│       ├── __init__.py
│       ├── data_loader.py   # Data loading utilities
│       ├── analysis.py      # Analysis functions
│       ├── visualization.py # Visualization utilities
│       └── main.py          # Main script with example workflow
├── tests/              # Test modules
├── .gitignore          # Git ignore file
├── pyproject.toml      # Project configuration
├── requirements.txt    # Production dependencies
├── requirements-dev.txt # Development dependencies
└── setup.py            # Package setup script
```

## Setup

1. Ensure you have Python 3.12.10 installed (preferably using pyenv)
2. Clone this repository
3. Set up the virtual environment:

```bash
# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .           # Install the package in development mode
pip install -r requirements-dev.txt  # Install development dependencies
```

## Usage

The main module provides an example workflow:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the main script
python -m src.insurance_analysis.main
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src tests
isort src tests
```

### Linting

```bash
flake8 src tests
mypy src tests