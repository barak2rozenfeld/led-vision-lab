# LED Vision Lab — Architecture

## 1. Architecture Goal

LED Vision Lab should be built as a modular system where dataset generation, training, inference, benchmarking, API, and web UI are separated from each other.

The main design principle is:

**All detection methods should expose the same interface.**

This allows the web application, REST API, benchmark framework, and automated tests to work with any detector without detector-specific logic.

---

## 2. High-Level Architecture

```text
                         ┌─────────────────────┐
                         │   Web Application   │
                         └─────────┬───────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │      REST API       │
                         │      FastAPI        │
                         └─────────┬───────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Detection Service   │
                         └─────────┬───────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
             HSV Detector     Template Detector    ML Detectors
                                                     │
                                              ┌──────┴──────┐
                                              ▼             ▼
                                             KNN           CNN
```

The training and dataset pipeline is separate:

```text
Synthetic Generator
        │
        ▼
     Dataset
        │
        ├────────────────────┐
        │                    │
        ▼                    ▼
 Feature Extraction      CNN Images
        │                    │
        ▼                    ▼
   KNN Training          CNN Training
        │                    │
        ▼                    ▼
    KNN Model            CNN Model
        │                    │
        └──────────┬─────────┘
                   ▼
             Model Storage
                   │
                   ▼
            Detection Service
```

---

## 3. Technology Stack

### Backend

Python

Primary responsibilities:

- Computer Vision
- Machine Learning
- Deep Learning
- Dataset generation
- Benchmarking
- REST API

---

### REST API

FastAPI

Responsibilities:

- Image upload
- Detector selection
- Detector comparison
- Request validation
- Error handling
- API documentation

---

### Computer Vision

OpenCV

Used for:

- Image loading
- ROI processing
- HSV conversion
- Color detection
- Image difference
- Template matching
- Image preprocessing
- Visualization

---

### Traditional Machine Learning

scikit-learn

Used initially for:

- KNN
- Feature preprocessing
- Metrics
- Train/test evaluation

---

### Deep Learning

PyTorch

Used for:

- Small CNN implementation
- Training
- Model loading
- Inference

The CNN should remain lightweight and CPU-compatible.

---

### Testing

pytest

Testing layers:

- Unit tests
- Detector tests
- Training utility tests
- API tests
- Integration tests

---

### Frontend

The frontend technology will be selected later.

Initial candidates:

- React / Next.js
- Simple server-rendered interface

The frontend should consume the REST API and must not contain Computer Vision logic.

---

## 4. Repository Structure

Target structure:

```text
led-vision-lab/
│
├── README.md
├── GOAL.md
├── ROADMAP.md
├── ARCHITECTURE.md
├── AGENTS.md
├── CONTRIBUTING.md
├── .gitignore
│
├── src/
│   └── led_vision/
│       │
│       ├── detectors/
│       │   ├── base.py
│       │   ├── hsv.py
│       │   ├── template.py
│       │   ├── knn.py
│       │   └── cnn.py
│       │
│       ├── preprocessing/
│       │   ├── image.py
│       │   └── roi.py
│       │
│       ├── features/
│       │   └── image_features.py
│       │
│       ├── models/
│       │
│       ├── synthetic/
│       │   ├── generator.py
│       │   └── config.py
│       │
│       ├── training/
│       │   ├── train_knn.py
│       │   └── train_cnn.py
│       │
│       ├── benchmark/
│       │   └── runner.py
│       │
│       └── api/
│           ├── app.py
│           └── routes/
│
├── configs/
│   ├── dataset/
│   ├── detectors/
│   └── training/
│
├── data/
│   ├── samples/
│   └── README.md
│
├── artifacts/
│   ├── models/
│   └── benchmarks/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── api/
│
├── web/
│
└── scripts/
```

This structure may evolve as the project grows.

Architecture changes should be intentional and documented.

---

## 5. Common Detector Interface

All detection methods must implement a common interface.

Conceptually:

```python
from abc import ABC, abstractmethod

class LedDetector(ABC):

    @abstractmethod
    def detect(self, image):
        pass
```

Every detector should return the same result structure.

Example:

```python
DetectionResult(
    state="ON",
    confidence=0.94,
    runtime_ms=3.2,
    method="hsv"
)
```

Recommended fields:

- state
- confidence
- runtime_ms
- method
- debug information
- optional visualization outputs

This abstraction is one of the most important architectural decisions in the project.

---

## 6. Detection Service

The Detection Service is responsible for selecting and running detectors.

Conceptual usage:

```python
service.detect(
    image=image,
    detector="hsv"
)
```

Or:

```python
service.compare(image)
```

The service should not contain the actual detection algorithms.

Instead, it delegates work to registered detector implementations.

Example:

```text
DetectionService
      │
      ├── hsv
      ├── template
      ├── knn
      └── cnn
```

---

## 7. Detector Registry

Detectors should be registered centrally.

Conceptual example:

```python
detectors = {
    "hsv": HSVDetector(),
    "template": TemplateDetector(),
    "knn": KNNDetector(),
    "cnn": CNNDetector(),
}
```

This enables future detectors to be added without changing API logic.

Potential future detectors:

- SVM
- Random Forest
- Vision Transformer
- External Vision API

---

## 8. Dataset Architecture

The complete generated dataset should not be stored inside Git.

Git should contain:

- Generator source code
- Dataset configuration
- Metadata schema
- Small sample images

Generated data should be stored locally or in external storage.

Example:

```text
data/
├── samples/
│
└── generated/
    ├── train/
    │   ├── on/
    │   └── off/
    │
    ├── validation/
    │   ├── on/
    │   └── off/
    │
    └── test/
        ├── on/
        └── off/
```

`data/generated/` should be ignored by Git.

---

## 9. Dataset Configuration

Dataset generation should be configuration-driven.

Example:

```yaml
seed: 42

samples: 1000

image:
  width: 128
  height: 128

states:
  on: 0.5
  off: 0.5

variation:
  brightness: true
  blur: true
  noise: true
  position: true
  size: true
```

This enables reproducible datasets and experiments.

---

## 10. Training Architecture

Training code should remain separate from inference code.

Example:

```text
Dataset
   │
   ▼
Training Pipeline
   │
   ▼
Model Artifact
   │
   ▼
Inference Detector
```

A detector should load a trained model.

It should not train a model during normal API operation.

---

## 11. Model Artifacts

Trained models should be treated as generated artifacts.

Examples:

```text
artifacts/models/
    knn_v001.pkl
    cnn_v001.pt
```

Each model should have associated metadata.

Example:

```json
{
  "model": "cnn_v001",
  "dataset": "dataset_v001",
  "git_commit": "abc123",
  "image_size": 64,
  "accuracy": 0.971,
  "seed": 42
}
```

Large model files should not automatically be committed to Git.

---

## 12. Benchmark Architecture

All detectors should be tested using the same benchmark runner.

```text
                Benchmark Dataset
                       │
                       ▼
                 Benchmark Runner
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
      HSV          Template           KNN
                                        │
                                        ▼
                                       CNN
                       │
                       ▼
                  Results
                       │
              ┌────────┴────────┐
              ▼                 ▼
             JSON              CSV
```

Metrics should include:

- Accuracy
- Precision
- Recall
- F1
- False ON
- False OFF
- Average inference latency

---

## 13. REST API Architecture

Initial API:

```text
GET /api/v1/health
```

```text
POST /api/v1/detect
```

```text
POST /api/v1/compare
```

Example flow:

```text
POST image
    │
    ▼
FastAPI
    │
    ▼
Image Validation
    │
    ▼
Detection Service
    │
    ▼
Selected Detector
    │
    ▼
DetectionResult
    │
    ▼
JSON Response
```

---

## 14. Web Architecture

The web application should only communicate with the API.

```text
Browser
   │
   ▼
Web Application
   │
   ▼
REST API
   │
   ▼
Detection Service
```

Do not implement detector logic in JavaScript.

This ensures that:

- Web
- CLI
- Automated tests
- External customers

can all use the same backend engine.

---

## 15. Explainability Outputs

Detectors may return optional debug information.

### HSV

Possible output:

- ROI
- HSV mask
- brightness score
- number of detected pixels

### Template

Possible output:

- reference image
- difference image
- similarity score

### KNN

Possible output:

- extracted features
- nearest-neighbor distances

### CNN

Possible output:

- ON probability
- OFF probability

These outputs can later be presented by the web UI.

---

## 16. Configuration

Hard-coded thresholds and model paths should be avoided.

Configuration should be external where practical.

Examples:

```text
configs/detectors/hsv.yaml
configs/training/cnn.yaml
configs/dataset/default.yaml
```

---

## 17. Logging

Important operations should produce structured logs.

Examples:

- detector selected
- runtime
- prediction
- error
- model version

Do not log uploaded image contents by default.

---

## 18. Error Handling

The system should provide clear errors for:

- Unsupported image formats
- Corrupted images
- Unknown detector
- Missing trained model
- Invalid configuration
- Internal inference failure

API errors should return structured responses.

---

## 19. Security Principles

The initial project is a research/demo platform, but basic security practices should be followed.

Examples:

- Validate uploaded files
- Limit upload size
- Do not execute uploaded content
- Do not expose internal file paths
- Store secrets outside Git
- Use environment variables for credentials

---

## 20. Architecture Principles

### Separation of Concerns

Dataset generation, training, detection, API, and UI should remain separate.

### Reproducibility

Dataset generation and training should be reproducible.

### Simplicity

Avoid unnecessary frameworks and abstractions.

### Extensibility

Adding a new detector should require minimal changes outside the detector itself.

### Testability

Core logic should be testable without running the web server.

### Explainability

Where possible, detectors should expose information explaining their decision.

---

## 21. Initial Implementation Order

Architecture should be implemented gradually.

Start with:

```text
Project Structure
      ↓
Common Detector Interface
      ↓
Synthetic Generator
      ↓
HSV Detector
```

Do not create the entire architecture in code immediately.

Only introduce components when their Roadmap phase begins.