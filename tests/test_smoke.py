"""Smoke test — verifies the test runner and package import work correctly."""


def test_package_importable() -> None:
    """The led_vision package must be importable."""
    import led_vision  # noqa: F401


def test_true() -> None:
    """Baseline: test runner is functional."""
    assert True
