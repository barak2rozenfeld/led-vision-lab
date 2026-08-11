# LED Vision Lab — AI Agent Instructions

## 1. Project Context

This repository contains **LED Vision Lab**.

The project compares multiple approaches for detecting whether an LED is ON or OFF from an image.

Before making significant changes, read:

1. `GOAL.md`
2. `ROADMAP.md`
3. `ARCHITECTURE.md`

These documents define the project direction, development order, and architecture.

---

## 2. Work Scope

Work on one clearly defined task at a time.

Do not implement future Roadmap phases unless explicitly requested.

Do not create large amounts of code "for later use."

Prefer incremental implementation.

Example:

If the current task is:

> Implement the HSV detector

Do not also create:

- CNN training
- authentication
- web UI
- Docker deployment
- database infrastructure

unless explicitly requested.

---

## 3. Architecture Rules

Respect the architecture defined in `ARCHITECTURE.md`.

Maintain separation between:

- Dataset generation
- Image preprocessing
- Detection
- Training
- Benchmarking
- REST API
- Web UI

Do not place detector logic inside API routes or frontend code.

Do not place training logic inside inference code.

---

## 4. Detector Interface

All detection methods must follow the common detector interface.

Supported initial detectors:

- HSV
- Template / Image Difference
- KNN
- CNN

The API, benchmark system, and web application should interact with detectors through the common interface.

Avoid detector-specific logic outside the detector implementation unless necessary.

---

## 5. Keep Solutions Simple

Prefer the simplest solution that satisfies the current requirement.

Avoid:

- unnecessary abstraction
- unnecessary inheritance
- premature optimization
- excessive design patterns
- large frameworks for small problems
- duplicated logic

Readable code is preferred over clever code.

---

## 6. Python Guidelines

Use modern Python.

Prefer:

- Type hints
- Small functions
- Clear names
- Dataclasses where useful
- Explicit error handling
- `pathlib` for filesystem paths
- Standard library solutions when practical

Avoid global state where possible.

Functions and classes should have one clear responsibility.

---

## 7. Configuration

Avoid hard-coded values when the value is likely to change.

Examples:

- HSV thresholds
- Dataset size
- Random seed
- Model path
- Image dimensions
- Training parameters

Use configuration files or clearly defined configuration objects.

Do not over-engineer configuration for constants that are truly fixed.

---

## 8. Dataset Rules

Do not commit the complete generated dataset to Git.

The repository should contain:

- Dataset generator code
- Configuration files
- Metadata schema
- Small sample images
- Documentation

Generated datasets should be reproducible.

Use deterministic random seeds where appropriate.

Generated data should normally be stored under an ignored directory such as:

```text
data/generated/
```

---

## 9. Synthetic Data

Synthetic data generation should support controlled variation.

Examples:

- ON / OFF state
- LED color
- brightness
- size
- position
- background
- ambient light
- exposure
- blur
- image noise
- camera angle

Each generated image should have enough metadata to reproduce or analyze the sample.

Do not generate 100,000 images during development by default.

Start with a small dataset for validation.

Example:

```text
100
500
1000
```

Scale to the full dataset only after visual and automated validation.

---

## 10. Machine Learning Experiments

Training must be reproducible.

Where applicable, record:

- Dataset version
- Dataset configuration
- Random seed
- Model configuration
- Hyperparameters
- Training metrics
- Validation metrics
- Git commit
- Timestamp

Do not overwrite previous experiment results unless explicitly requested.

---

## 11. Model Artifacts

Trained models are generated artifacts.

Examples:

```text
cnn_v001.pt
knn_v001.pkl
```

Do not automatically commit large trained models to Git.

Model metadata should identify:

- Model type
- Model version
- Dataset version
- Configuration
- Metrics

---

## 12. Testing

Every meaningful feature should include appropriate tests.

Prefer:

- Unit tests for pure logic
- Integration tests for component interaction
- API tests for HTTP endpoints

Before considering a task complete:

1. Run relevant tests
2. Confirm existing tests still pass
3. Check common error cases
4. Report what was tested

Do not claim tests passed unless they were actually executed.

---

## 13. Benchmarking

All detectors should be benchmarked against the same test dataset when comparisons are made.

Important metrics include:

- Accuracy
- Precision
- Recall
- F1 score
- False ON rate
- False OFF rate
- Inference runtime

Do not present benchmark results from different datasets as a fair detector comparison.

---

## 14. API Rules

The REST API should delegate detection to the Detection Service.

API routes should mainly handle:

- Request parsing
- Validation
- Calling application services
- Error conversion
- Response serialization

Do not implement Computer Vision algorithms directly inside FastAPI routes.

---

## 15. Frontend Rules

The frontend should consume the REST API.

Do not duplicate Computer Vision or ML logic in the frontend.

The frontend is responsible for:

- Image upload
- Detector selection
- Result presentation
- Comparison presentation
- Educational explanations

---

## 16. Dependencies

Do not add a dependency unless it provides clear value.

Before adding a dependency:

1. Check whether the project already has a suitable solution
2. Prefer the standard library when practical
3. Keep dependencies maintained and widely used
4. Explain the reason for introducing it

Update dependency files when adding or removing packages.

---

## 17. Documentation

Update documentation when a change affects:

- Public API
- Architecture
- Dataset format
- Configuration
- Setup procedure
- Model behavior
- Project structure

Do not let documentation knowingly become inconsistent with the implementation.

---

## 18. Git Workflow

Prefer small and focused changes.

One task should normally correspond to one branch / Pull Request.

Suggested branch naming:

```text
feature/12-hsv-detector
feature/18-synthetic-generator
fix/31-image-validation
docs/7-update-architecture
```

Suggested commit format:

```text
feat(hsv): implement LED state detection
test(hsv): add detector unit tests
fix(api): reject invalid image uploads
docs(dataset): document metadata schema
refactor(core): simplify detector registry
```

Avoid commits such as:

```text
update
changes
fix
working
stuff
```

---

## 19. Do Not Modify Unrelated Code

When fixing a bug or implementing a feature:

- Avoid unrelated refactoring
- Avoid renaming unrelated files
- Avoid changing formatting across the entire project
- Avoid altering APIs that are outside the task

If an unrelated problem is discovered, report it separately.

---

## 20. Error Handling

Do not silently ignore errors.

Provide clear errors for conditions such as:

- Invalid image
- Unsupported format
- Missing configuration
- Missing model
- Unknown detector
- Failed inference

Errors should help the developer understand what went wrong.

---

## 21. Security

Treat uploaded images as untrusted input.

Follow basic security practices:

- Validate file type
- Limit image size
- Do not execute uploaded content
- Do not expose internal filesystem paths
- Do not commit secrets
- Use environment variables for credentials

---

## 22. Performance

Measure before optimizing.

Do not introduce complexity for assumed performance problems.

For detector execution, record inference time where practical.

Performance improvements should preserve correctness and test coverage.

---

## 23. Before Starting a Task

Before implementing a requested feature:

1. Identify the relevant Roadmap phase
2. Inspect the existing implementation
3. Verify the architecture expectations
4. Determine the smallest useful implementation
5. Identify tests required

If the requested change conflicts with `GOAL.md`, `ROADMAP.md`, or `ARCHITECTURE.md`, clearly explain the conflict before making a major architectural change.

---

## 24. Definition of Done

A development task is complete when:

- The requested behavior is implemented
- Code follows the project architecture
- Relevant tests exist
- Relevant tests pass
- Error cases are handled
- Documentation is updated if required
- No unrelated changes were introduced

---

## 25. Agent Response Style

When completing development work, provide a concise summary containing:

- What changed
- Which files changed
- Tests executed
- Any important assumptions
- Suggested next step

Do not provide long explanations unless requested.

---

## Core Principle

Build LED Vision Lab incrementally.

**Correctness and clarity are more important than generating large amounts of code quickly.**