"""Pytest configuration for checkout-local temporary directories."""

import os
import time
from pathlib import Path
from uuid import uuid4

import pytest


def _selected_repository_root() -> Path:
    """Return outer SynthData root or nearest standalone SynthEval checkout."""
    conftest_path = Path(__file__).resolve()
    candidates = tuple(conftest_path.parents)

    synthdata_roots = [
        candidate.resolve()
        for candidate in candidates
        if (candidate.resolve() / "synthdata" / "__init__.py").is_file()
        and (candidate.resolve() / "pyproject.toml").is_file()
    ]
    if synthdata_roots:
        return synthdata_roots[-1]

    checkout_roots = [
        candidate
        for candidate in candidates
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").is_file()
    ]
    if checkout_roots:
        return checkout_roots[0]

    raise RuntimeError(
        "Unable to locate SynthEval checkout root from "
        f"{conftest_path}: required checkout markers are absent"
    )


def _reject_symlink_components(path: Path) -> None:
    """Reject scratch path components that could redirect pytest outside checkout."""
    current = Path(path.anchor) if path.is_absolute() else Path()
    for component in path.parts[1:] if path.is_absolute() else path.parts:
        current /= component
        if current.is_symlink():
            raise RuntimeError(f"Refusing symlink in pytest scratch path: {current}")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Set unique local basetemp before pytest temporary fixtures are used."""
    repository_root = _selected_repository_root().resolve()
    scratch_root = repository_root / "tmp" / "pytest"
    _reject_symlink_components(scratch_root)
    scratch_root = scratch_root.resolve()
    run_id = f"pytest-{os.getpid()}-{time.time_ns()}-{uuid4().hex}"
    basetemp = scratch_root / run_id
    _reject_symlink_components(basetemp)
    basetemp = basetemp.resolve()

    if not scratch_root.is_relative_to(repository_root):
        raise RuntimeError(f"Refusing pytest scratch outside checkout root: {scratch_root}")
    if not basetemp.is_absolute() or not basetemp.is_relative_to(scratch_root):
        raise RuntimeError(f"Refusing pytest basetemp outside checkout scratch: {basetemp}")

    try:
        scratch_root.mkdir(parents=True, exist_ok=True)
        basetemp.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise RuntimeError(f"Unable to create pytest scratch directory: {basetemp}") from exc

    config.option.basetemp = str(basetemp)
