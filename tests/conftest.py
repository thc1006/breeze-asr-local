"""Shared pytest fixtures: synthetic audio + path to the real user-supplied m4a."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow integration tests (real model download + transcribe)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: integration tests that need the real 1.6 GB model"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_SAMPLE_M4A = PROJECT_ROOT / "1woman.m4a"


def _sine(duration_s: float, sample_rate: int, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def wav_44k_stereo(tmp_path: Path) -> Path:
    """Synthetic 2 s, 44.1 kHz stereo WAV — exercises resampling + downmix paths."""
    path = tmp_path / "sine_44k_stereo.wav"
    mono = _sine(2.0, 44100)
    stereo = np.column_stack([mono, mono])
    sf.write(str(path), stereo, 44100, subtype="PCM_16")
    return path


@pytest.fixture
def wav_16k_mono(tmp_path: Path) -> Path:
    """Already target-format WAV (16 kHz mono PCM_16)."""
    path = tmp_path / "sine_16k_mono.wav"
    sf.write(str(path), _sine(1.5, 16000), 16000, subtype="PCM_16")
    return path


@pytest.fixture
def real_m4a() -> Path:
    """The user-supplied test clip. Skip dependent tests if not present."""
    if not REAL_SAMPLE_M4A.exists():
        pytest.skip(f"real sample {REAL_SAMPLE_M4A} not present")
    return REAL_SAMPLE_M4A
