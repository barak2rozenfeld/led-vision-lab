# LED Vision Lab

A computer vision platform that detects whether an LED is **ON** or **OFF** from an image, comparing multiple detection approaches to find the best trade-off for different environments.

## Problem

In hardware testing and manufacturing, engineers need to automatically verify visual indicators (LEDs):

```
Test System → Send command to DUT → LED should turn ON
→ Camera captures image → Vision system detects state → PASS / FAIL
```

## Detection Methods

| Method | Type | Training Required |
|--------|------|:-----------------:|
| HSV / Color Detection | Classical CV | No |
| Template / Image Difference | Classical CV | No |
| KNN Classifier | Traditional ML | Yes |
| Small CNN | Deep Learning | Yes |

All methods expose the same interface and are evaluated on the same benchmark dataset.

## Project Structure

```
led-vision-lab/
├── src/led_vision/       # Core Python package
│   ├── detectors/        # HSV, Template, KNN, CNN detectors
│   ├── preprocessing/    # Image loading and ROI extraction
│   ├── features/         # Feature extraction for ML detectors
│   ├── synthetic/        # Synthetic dataset generator
│   ├── training/         # KNN and CNN training pipelines
│   ├── benchmark/        # Benchmark runner
│   └── api/              # FastAPI REST API
├── configs/              # YAML configuration files
├── data/                 # Sample images (generated data is gitignored)
├── artifacts/            # Trained models and benchmark results
├── tests/                # Unit, integration, and API tests
├── scripts/              # Utility scripts
└── web/                  # Web application (Phase 9)
```

## Setup

### Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)

### Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Create environment and install dependencies

```bash
uv sync --extra dev
```

### Activate virtual environment

```powershell
# Windows
.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

## Running Tests

```bash
pytest
```

## Generating a Dataset

> Available from Phase 1.

```bash
python scripts/generate_dataset.py --config configs/dataset/default.yaml
```

## Running the API

> Available from Phase 8.

```bash
uvicorn led_vision.api.app:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/detect` | Detect LED state |
| POST | `/api/v1/compare` | Compare all detectors |

## Core Research Question

> What is the simplest, fastest, and most reliable method for detecting whether an LED is ON or OFF under different visual conditions?

## Documentation

- [`Goal.md`](Goal.md) — Project vision and objectives
- [`Roadmap.md`](Roadmap.md) — Development phases
- [`Architecture.md`](Architecture.md) — System design
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — Contribution guidelines
