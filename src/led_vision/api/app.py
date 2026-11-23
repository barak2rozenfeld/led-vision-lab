"""FastAPI application factory.

Run from the repository root:
    uv run uvicorn led_vision.api.app:create_default_app --factory --port 8000
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from led_vision.detectors.registry import build_detectors
from led_vision.detectors.service import DetectionService

from .config import DEFAULT_API_CONFIG, ApiConfig, load_api_config
from .routes.v1 import API_VERSION, router


def create_app(config: ApiConfig, service: DetectionService | None = None) -> FastAPI:
    app = FastAPI(
        title="LED Vision Lab API",
        description="Detect whether an LED is ON or OFF with interchangeable methods.",
        version=API_VERSION,
    )
    app.state.config = config
    app.state.service = service or DetectionService(build_detectors(config.detector_config_dir))
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=list(config.cors_origins),
            allow_methods=["GET", "POST"], allow_headers=["*"],
        )
    app.include_router(router)
    return app


def create_default_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app(load_api_config(DEFAULT_API_CONFIG))
