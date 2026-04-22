# MLOps Batch Job – Rolling-Mean Signal Pipeline

A minimal, reproducible MLOps-style batch job that generates a binary trading signal from OHLCV data using a rolling mean strategy.

## How it works

1. Loads config from YAML (seed, window, version)
2. Reads OHLCV CSV, validates structure
3. Computes rolling mean on `close` (configurable window)
4. Generates signal: `1` if `close > rolling_mean`, else `0`
5. Writes structured `metrics.json` and detailed `run.log`

The first `window - 1` rows are excluded from signal computation (NaN rolling-mean rows).

---

## Local run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

### Outputs

| File | Description |
|------|-------------|
| `metrics.json` | Machine-readable metrics (signal_rate, latency, etc.) |
| `run.log` | Detailed timestamped log |

---

## Docker

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

The container prints the final `metrics.json` to stdout and exits with code `0` on success, non-zero on failure.

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4990,
  "latency_ms": 127,
  "seed": 42,
  "status": "success"
}
```

> `rows_processed` is 9996 (not 10000) because the first `window - 1 = 4` rows have no valid rolling mean and are excluded from signal computation.

---

## Config reference

```yaml
seed: 42       # NumPy random seed for reproducibility
window: 5      # Rolling-mean window size
version: "v1"  # Pipeline version tag written to metrics
```

---

## Error output

If any validation or processing step fails, `metrics.json` will contain:

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

Exit code will be non-zero.
