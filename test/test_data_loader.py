def test_config_loading():
    """
    Ensure the scheduling configuration module exposes the expected symbols
    and default values.
    """
    import importlib

    cfg = importlib.import_module(
        "scheduling.config.scheduling_config"
    )

    # Basic type checks
    assert isinstance(cfg.WORKLOAD_PROFILE, dict)
    assert isinstance(cfg.SERVER_CONFIG, dict)
    assert isinstance(cfg.SERVER_POWER_MODEL, dict)

    # Value checks
    assert cfg.WORKLOAD_PROFILE["cpu_requirement"] == 0.25
    assert cfg.SERVER_CONFIG["cores"] == 8
    assert cfg.RESAMPLE_STRATEGY in {
        "upsample_solar",
        "downsample_cpu",
        "common_5min",
    }
    assert cfg.TARGET_FREQ == "5T"
    assert cfg.LOG_LEVEL in {"INFO", "DEBUG"}

import pickle
from pathlib import Path

import pandas as pd
import pytest

from scheduling.src.data_loader import load_green_energy, align_timeseries


def test_load_green_energy(tmp_path: Path):
    """Green-energy loader should correctly unpickle a pandas Series."""
    series = pd.Series(range(10), index=pd.date_range("2023-01-01", periods=10, freq="15T"))
    obj_path = tmp_path / "solar.obj"
    with obj_path.open("wb") as fh:
        pickle.dump(series, fh)

    loaded = load_green_energy(obj_path)
    pd.testing.assert_series_equal(series, loaded)


def test_align_timeseries_invalid_strategy():
    """Supplying an unknown strategy must raise ValueError."""
    cpu = {"m1": pd.Series(range(3), index=pd.date_range("2023-01-01", periods=3, freq="5T"))}
    solar = pd.Series(range(1), index=pd.date_range("2023-01-01", periods=1, freq="15T"))
    with pytest.raises(ValueError):
        align_timeseries(cpu, solar, strategy="not_a_strategy")