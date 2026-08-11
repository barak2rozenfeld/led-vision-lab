# Data Directory

## Structure

```
data/
├── samples/          # Small hand-picked sample images (committed to Git)
└── generated/        # Auto-generated synthetic dataset (gitignored — do not commit)
    ├── train/
    │   ├── on/
    │   └── off/
    ├── validation/
    │   ├── on/
    │   └── off/
    └── test/
        ├── on/
        └── off/
```

## Generating the Dataset

> Available from Phase 1.

```bash
python scripts/generate_dataset.py --config configs/dataset/default.yaml
```

Generated images are stored under `data/generated/` which is excluded from Git via `.gitignore`.

## Reproducing a Dataset

Every generated dataset is reproducible using its configuration file and random seed.

The seed and all generation parameters are recorded in `data/generated/metadata.json`.
