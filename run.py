"""
MLOps batch job: rolling-mean signal pipeline.

Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler – detailed
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler – INFO+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    with path.open() as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    # Type checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"'version' must be a non-empty string, got: {cfg['version']!r}")

    logger.info(
        "Config loaded – version=%s  seed=%d  window=%d",
        cfg["version"], cfg["seed"], cfg["window"],
    )
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV '{input_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset is empty (no data rows): {input_path}")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Columns present: {list(df.columns)}"
        )

    non_numeric = df["close"].apply(lambda x: not isinstance(x, (int, float, np.integer, np.floating)))
    if non_numeric.any():
        raise ValueError("Column 'close' contains non-numeric values.")

    if df["close"].isna().all():
        raise ValueError("Column 'close' contains only NaN values.")

    logger.info("Dataset loaded – rows=%d  columns=%s", len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    """
    Compute rolling mean on 'close'.

    The first (window - 1) rows will be NaN; those rows are excluded from
    signal computation. This is the standard pandas rolling-mean behaviour
    with min_periods=window.
    """
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    valid = rolling_mean.notna().sum()
    logger.info(
        "Rolling mean computed – window=%d  valid_rows=%d  nan_rows=%d",
        window, valid, len(df) - valid,
    )
    return rolling_mean


def generate_signals(close: pd.Series, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    """
    signal = 1 if close > rolling_mean else 0.
    Rows where rolling_mean is NaN are excluded (signal stays NaN → not counted).
    """
    mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=close.index)
    signal[mask] = (close[mask] > rolling_mean[mask]).astype(int)

    valid_count = mask.sum()
    signal_count = signal[mask].sum()
    logger.info(
        "Signals generated – valid_rows=%d  signal_1_count=%d  signal_0_count=%d",
        valid_count, int(signal_count), int(valid_count - signal_count),
    )
    return signal


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps rolling-mean signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV")
    parser.add_argument("--config",   required=True, help="Path to YAML config")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file", help="Path for log file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    t_start = time.time()
    logger.info("=== Job started ===")
    logger.info(
        "Args – input=%s  config=%s  output=%s  log_file=%s",
        args.input, args.config, args.output, args.log_file,
    )

    version = "unknown"

    try:
        # 1. Config
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        seed: int    = cfg["seed"]
        window: int  = cfg["window"]

        np.random.seed(seed)
        logger.debug("NumPy random seed set to %d", seed)

        # 2. Dataset
        df = load_dataset(args.input, logger)

        # 3. Rolling mean
        logger.info("Computing rolling mean (window=%d) …", window)
        rolling_mean = compute_rolling_mean(df, window, logger)

        # 4. Signals
        logger.info("Generating binary signals …")
        signal = generate_signals(df["close"], rolling_mean, logger)

        # 5. Metrics
        valid_mask     = signal.notna()
        rows_processed = int(valid_mask.sum())
        signal_rate    = float(signal[valid_mask].mean())
        latency_ms     = int((time.time() - t_start) * 1000)

        logger.info(
            "Metrics – rows_processed=%d  signal_rate=%.4f  latency_ms=%d",
            rows_processed, signal_rate, latency_ms,
        )

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          round(signal_rate, 4),
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        write_metrics(args.output, metrics)
        logger.info("Metrics written to %s", args.output)
        logger.info("=== Job finished – status=success ===")

        # Print final metrics to stdout (required for Docker run)
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.time() - t_start) * 1000)
        error_msg  = str(exc)
        logger.error("Job failed: %s", error_msg, exc_info=True)
        logger.info("=== Job finished – status=error ===")

        error_metrics = {
            "version":       version,
            "status":        "error",
            "error_message": error_msg,
        }
        try:
            write_metrics(args.output, error_metrics)
            logger.info("Error metrics written to %s", args.output)
        except Exception as write_exc:  # noqa: BLE001
            logger.error("Could not write error metrics: %s", write_exc)

        print(json.dumps(error_metrics, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
