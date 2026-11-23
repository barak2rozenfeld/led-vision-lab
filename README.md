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

## Synthetic Dataset

LED images are generated procedurally: 14 LED colors, 6 shapes (round, square, rectangular, dome, side-view dome, SMD), 15 background styles, ambient lighting, glare, occlusion, camera noise/blur, false-positive distractors, and four difficulty levels. Every image has a 40-field metadata row (`metadata.csv`) and every run is reproducible from its seed.

![Sample grid — one ON and one OFF example per background](docs/images/sample_grid.jpg)

| Config | Scenes | Export sizes (px) | Images | Focus |
|--------|-------:|-------------------|-------:|-------|
| `configs/dataset/default.yaml` | 100 | 64, 128, 224, 256 | 400 | Quick smoke run |
| `configs/dataset/large_scale.yaml` | 50,000 | 64, 128, 224, 256 | 200,000 | General mix |
| `configs/dataset/pcb_1m.yaml` | 250,000 | 64, 128, 224, 256 | 1,000,000 | ~65 % PCB backgrounds |
| `configs/dataset/high_res.yaml` | 10,000 | 512, 1024, 2048 | 30,000 | High-resolution |

Generated datasets are not committed. Small samples live in [`data/samples/images`](data/samples/images).

```bash
uv run python scripts/generate_dataset.py --config configs/dataset/pcb_1m.yaml
uv run python scripts/make_samples.py --dataset data/generated_pcb_1m   # refresh samples + grid
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
