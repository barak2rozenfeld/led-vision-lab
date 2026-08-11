# LED Vision Lab — Roadmap

## Project Objective

Build a complete LED state detection platform that compares classical Computer Vision, Machine Learning, and Deep Learning methods.

The project should support:

* Synthetic dataset generation
* LED ON/OFF detection
* Multiple detection approaches
* Benchmarking and comparison
* REST API
* Web application
* Documentation
* Potential commercial use

---

# Phase 0 — Project Foundation

## Goal

Create a clean and maintainable project structure before implementing algorithms.

## Tasks

* Initialize Git repository
* Create project documentation
* Define architecture
* Define coding conventions
* Define AI agent instructions
* Create `.gitignore`
* Create Python virtual environment
* Define dependency management
* Create base folder structure
* Configure basic testing framework

## Deliverables

* `README.md`
* `GOAL.md`
* `ROADMAP.md`
* `ARCHITECTURE.md`
* `AGENTS.md`
* Working Python environment
* when working on website backend use Node js, and other
*for Beautifull website webapp use make it WOW Affecte and much much more beautiful
* Initial repository structure

## Done When

* Project can be cloned and opened cleanly
* Python environment can be created from documented instructions
* Basic test command runs successfully

---

# Phase 1 — Synthetic Dataset Generator

## Goal

Create a reproducible synthetic LED dataset without requiring real hardware.

## Initial Target

Generate up to:

**100,000 labeled LED images**

Primary classification:

* ON
* OFF

## Variations

Dataset generation should support variations in:

* LED brightness
* LED size
* LED position
* LED color
* Ambient lighting
* Background
* Camera angle
* Camera distance
* Exposure
* Blur
* Noise
* Contrast
* Reflections

## Metadata

Each generated image should have associated metadata.

Example fields:

* image ID
* state
* color
* brightness
* background
* noise level
* blur level
* camera angle
* random seed

## Important Principle

Generated images should not be manually committed to Git.

The repository should contain:

* generator code
* configuration
* metadata schema
* a small sample dataset
* reproduction instructions

## Done When

* Dataset generation works from one command
* Generation is reproducible using a random seed
* ON/OFF labels are generated automatically
* Metadata is generated automatically
* At least a small validation dataset can be visually inspected
* Generator can scale toward 100K images

---

# Phase 2 — Common Detection Interface

## Goal

Create one common interface that all LED detection methods implement.

Example concept:

```python
result = detector.detect(image)
```

Common result format:

```json
{
  "state": "ON",
  "confidence": 0.92,
  "runtime_ms": 3.4,
  "method": "hsv"
}
```

## Purpose

HSV, Template Matching, KNN, and CNN should all be interchangeable from the perspective of the API and web application.

## Done When

* Common detector interface exists
* Common result schema exists
* Detectors can be registered and selected by name
* Unit tests exist for the interface

---

# Phase 3 — HSV / Color Detection

## Goal

Implement the simplest non-ML baseline.

## Features

* ROI processing
* RGB to HSV conversion
* Brightness analysis
* Saturation analysis
* Color thresholding
* Configurable thresholds
* Visualization of generated mask

## Outputs

* ON / OFF
* Detection score
* Runtime
* HSV mask for explainability

## Done When

* HSV detector runs through the common interface
* Automated tests exist
* Performance is measured on the benchmark dataset
* Example visualizations are available

---

# Phase 4 — Template / Image Difference

## Goal

Implement a reference-based Computer Vision detector.

## Techniques To Evaluate

* Absolute image difference
* Template matching
* Similarity score

Optional later extension:

* SSIM

## Outputs

* ON / OFF
* Similarity / difference score
* Runtime
* Difference visualization

## Done When

* Detector works through the common interface
* Reference management exists
* Tests exist
* Benchmark results can be generated

---

# Phase 5 — KNN Classifier

## Goal

Implement a traditional Machine Learning baseline.

## Feature Extraction

Possible features:

* Mean brightness
* Maximum brightness
* Standard deviation
* RGB averages
* HSV averages
* Saturation
* Contrast
* Histogram features

## Training

Create:

* training pipeline
* validation split
* saved model
* inference implementation

## Experimentation

Evaluate different values of K.

## Done When

* Feature extractor exists
* KNN can be trained reproducibly
* Model can be saved and loaded
* KNN implements the common detector interface
* Metrics are generated
* Tests exist

---

# Phase 6 — Small CNN

## Goal

Train a lightweight CNN for LED ON/OFF classification.

## Requirements

The model should be:

* Small
* Fast
* CPU friendly
* Suitable for test automation environments

## Training Pipeline

Support:

* train / validation / test split
* configurable image resolution
* configurable batch size
* epochs
* learning rate
* checkpoint saving
* experiment metadata

## Metrics

Measure:

* Accuracy
* Precision
* Recall
* F1 score
* False ON rate
* False OFF rate
* Inference latency
* Model size

## Done When

* CNN can be trained end-to-end
* Model can be saved and loaded
* CNN implements the common detector interface
* Training results are reproducible
* Benchmark results are available

---

# Phase 7 — Benchmark Framework

## Goal

Compare all detection approaches using exactly the same test data.

## Detectors

* HSV
* Template / Difference
* KNN
* CNN

## Metrics

Compare:

* Accuracy
* Precision
* Recall
* F1
* False Positive Rate
* False Negative Rate
* Runtime
* Model size
* Explainability
* Training requirement

## Environmental Analysis

Evaluate performance under:

* low brightness
* high brightness
* blur
* noise
* different LED colors
* different camera angles
* different distances

## Deliverables

Generate machine-readable results and human-readable reports.

Example:

```text
benchmarks/results/
    benchmark_001.json
    benchmark_001.csv
```

## Done When

* One command runs the benchmark
* All detectors run against the same dataset
* Results are saved
* Comparison tables can be generated
* Results can later be displayed in the web application

---

# Phase 8 — REST API

## Goal

Expose LED detection as a service.

## Initial Endpoints

### Health

`GET /api/v1/health`

### Detect

`POST /api/v1/detect`

Inputs:

* image
* detector

### Compare

`POST /api/v1/compare`

Runs multiple detectors against the same image.

## Example Response

```json
{
  "state": "ON",
  "confidence": 0.97,
  "method": "cnn",
  "runtime_ms": 9.2
}
```

## Requirements

* Input validation
* Error handling
* Structured responses
* API documentation
* Automated API tests

## Done When

* Image can be uploaded over HTTP
* Detector can be selected
* Compare endpoint works
* API tests pass
* API documentation exists

---

# Phase 9 — Web Application

## Goal

Create a public-facing interface for the platform.

## Main Pages

### Home

Explain:

* Problem
* Project
* Supported methods

### Detect

User can:

* Upload image
* Select detector
* Run detection
* View result

### Compare

User can:

* Upload one image
* Run all detection methods
* Compare results

### Dataset

Show:

* Dataset statistics
* Generation methodology
* Example images
* Distribution

### Models

Explain:

* HSV
* Template Matching
* KNN
* CNN

For each method:

* How it works
* Advantages
* Disadvantages
* Speed
* Training requirements
* Recommended use cases

### Benchmarks

Display comparison results.

### API

Provide API documentation and examples.

## Done When

* User can upload an image
* User can select detector
* Results are shown clearly
* Compare mode works
* Model explanation page exists
* Benchmark results are presented

---

# Phase 10 — Production Quality

## Goal

Make the project look and behave like a real engineering product.

## Tasks

* CI pipeline
* Automated tests
* Code formatting
* Static analysis
* Docker support
* Configuration management
* Logging
* Error handling
* API rate limiting
* Security review
* Deployment documentation

## GitHub

Add:

* Pull Request template
* Issue templates
* GitHub Actions
* Release tags
* Changelog

## Done When

* CI runs automatically on Pull Requests
* Tests must pass before merge
* Application can run from documented setup steps
* Docker deployment works

---

# Phase 11 — Real-World Validation

## Goal

Measure the synthetic-to-real domain gap.

When real hardware becomes available:

* Capture real ON images
* Capture real OFF images
* Test all detectors
* Compare synthetic vs. real performance
* Fine-tune CNN if necessary
* Adjust synthetic generator based on observed differences

## Important

The final evaluation should clearly distinguish:

* Synthetic test accuracy
* Real-world test accuracy

## Done When

* Real test dataset exists
* Domain-gap analysis exists
* Final model performance is documented

---

# Phase 12 — Commercial Exploration

## Goal

Evaluate whether the project can become a useful service.

Potential offerings:

### Free Demo

Limited browser-based detections.

### Developer API

API key access with usage limits.

### Professional API

Higher limits and batch processing.

### On-Premise

Local deployment for:

* Test laboratories
* Manufacturing environments
* Secure companies
* Hardware validation teams

## Potential Future Features

* Authentication
* API keys
* Usage tracking
* Rate limits
* Billing
* Team accounts
* Private models
* Custom LED training
* On-premise deployment

Commercial features should be added only after the core detection system is proven.

---

# Final Project Deliverables

The completed project should include:

* Synthetic dataset generator
* 100K-image generation capability
* HSV detector
* Template detector
* KNN detector
* Small CNN
* Unified detector interface
* Benchmark framework
* REST API
* Web application
* Automated tests
* CI/CD
* Documentation
* Experiment results
* Final technical report
* Public demo or deployable package

---

# Recommended Development Order

```text
Foundation
    ↓
Synthetic Generator
    ↓
Common Detector Interface
    ↓
HSV
    ↓
Template
    ↓
KNN
    ↓
CNN
    ↓
Benchmark
    ↓
API
    ↓
Web
    ↓
Production Hardening
    ↓
Real-World Validation
    ↓
Commercial Features
```

# Core Principle

Do not choose the most complex solution by default.

The project should answer:

**Which technique provides the best trade-off between accuracy, speed, complexity, explainability, and deployment cost for LED state detection?**
