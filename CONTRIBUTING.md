# Contributing to LED Vision Lab

## Development Setup

```bash
uv sync --extra dev
```

## Branch Naming

```
feature/<issue>-<short-description>
fix/<issue>-<short-description>
docs/<issue>-<short-description>
refactor/<issue>-<short-description>
```

Examples:

```
feature/12-hsv-detector
feature/18-synthetic-generator
fix/31-image-validation
docs/7-update-architecture
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(hsv): implement LED state detection
test(hsv): add detector unit tests
fix(api): reject invalid image uploads
docs(dataset): document metadata schema
refactor(core): simplify detector registry
```

Avoid:

```
update
changes
fix
working
stuff
```

## Code Standards

- **Type hints** on all functions and class attributes
- **`pathlib.Path`** for all filesystem paths (not `os.path`)
- **Dataclasses** for structured data (`DetectionResult`, config objects)
- **Explicit error handling** — never silently swallow exceptions
- **Small functions** — one clear responsibility per function
- **No global state** where avoidable

Format code before committing:

```bash
black src/ tests/
ruff check src/ tests/ --fix
```

Type-check:

```bash
mypy src/
```

## Architecture Rules

- All detectors must implement the common `LedDetector` interface
- No CV or ML logic inside API routes or frontend code
- No training logic inside inference code
- Dataset generation, detection, training, API, and web UI stay separated

## Definition of Done

A task is done when:

- [ ] Requested behavior is implemented
- [ ] Code follows the project architecture
- [ ] Relevant tests exist and pass
- [ ] Error cases are handled
- [ ] Documentation updated if required
- [ ] No unrelated changes introduced

## Running Tests

```bash
pytest
pytest --cov=led_vision tests/
```
