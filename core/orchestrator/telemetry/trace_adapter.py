"""
orchestrator/telemetry/trace_adapter.py
─────────────────────────────────────────
AlibabaMachineTraceAdapter: replays Alibaba 2018 cluster trace data as per-node
telemetry, replacing the i.i.d. random.gauss() generator in TelemetryCollector.

Background
──────────
The Alibaba 2018 cluster trace (Zenodo record 14564935) provides aggregate CPU
and memory utilisation sampled every 300 seconds over 8 days (~2243 rows).
CPU spans 16–79% (mean ≈ 40%). Unlike Gaussian noise, the trace exhibits:
  - Temporal autocorrelation (avg tick-to-tick change: 4.6%)
  - Diurnal cycles (load rises/falls through each 24-hour period)
  - Burst events (sudden spikes, not just gradual drifts)

These properties give the LSTM predictor realistic patterns to learn from,
which makes spike_probability estimates more meaningful in tests.

Design
───────
The CSV contains a single aggregate timeseries (no per-machine column). Each of
the 5 mock nodes is mapped to:
  - A unique time offset into the trace (so nodes see different sections of the
    8-day window, making their telemetry independent)
  - A per-node cpu_scale and cpu_bias (so each node's average CPU matches its
    expected baseline — e.g. node-arm-02 targets ~20%, node-api-03 targets ~60%)

After the 8-day trace is exhausted, tick_number wraps around (circular buffer),
so tests can run indefinitely without IndexError.

Integration
────────────
    adapter = AlibabaMachineTraceAdapter("tests/fixtures/alibaba_machine_usage_300s.csv")
    collector = TelemetryCollector(svc, trace_adapter=adapter)
    collector.tick()   # now uses real trace data instead of random.gauss

inject_spike() still works transparently on top of trace replay — spike overrides
the trace CPU value with SPIKE_CPU_UTIL for the duration of the spike window.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Regex to extract cpus and memory from Borg average_usage strings.
# Matches: {'cpus': 0.021, 'memory': 0.014}  OR  {'cpus': 2.5e-05, 'memory': None}
_BORG_USAGE_RE = re.compile(
    r"'cpus':\s*([\d.e+\-]+).*?'memory':\s*([\d.e+\-]+|None)",
    re.DOTALL,
)


# ── Per-node configuration ──────────────────────────────────────────────────────

# Traces the single aggregate timeseries using a unique starting offset per node
# so each node sees a different section of the 8-day Alibaba trace.
#
# cpu_scale + cpu_bias transform the raw CPU value (16–79% range, mean 40%) to
# match each node's expected steady-state utilisation:
#
#   node-cpu-01: target ~35% → scale=0.85, bias=0.0  → mean ≈ 34%
#   node-arm-02: target ~20% → scale=0.50, bias=0.0  → mean ≈ 20%
#   node-api-03: target ~60% → scale=1.00, bias=19.8 → mean ≈ 60%
#   node-gpu-04: target ~45% → scale=1.00, bias=4.8  → mean ≈ 45%
#   node-gpu-05: target ~30% → scale=0.70, bias=2.0  → mean ≈ 30%
#
# Offsets are spaced evenly across the 2243-row trace to maximise independence.

_NODE_CONFIG: Dict[str, Dict[str, float]] = {
    "node-cpu-01": {"offset": 0,    "cpu_scale": 0.85, "cpu_bias":  0.0},
    "node-arm-02": {"offset": 448,  "cpu_scale": 0.50, "cpu_bias":  0.0},
    "node-api-03": {"offset": 896,  "cpu_scale": 1.00, "cpu_bias": 19.8},
    "node-gpu-04": {"offset": 1344, "cpu_scale": 1.00, "cpu_bias":  4.8},
    "node-gpu-05": {"offset": 1791, "cpu_scale": 0.70, "cpu_bias":  2.0},
}

# Scaling factor for memory. The raw trace memory is 78–95% (high, near-constant
# since cluster memory is heavily allocated). Scale to 39–48% to match our 50%
# baseline and avoid unrealistic memory pressure in tests.
_MEM_SCALE: float = 0.50

# Fallback values returned for any node_id not in _NODE_CONFIG.
_FALLBACK_CPU: float = 40.0
_FALLBACK_MEM: float = 50.0


class AlibabaMachineTraceAdapter:
    """
    Replay the Alibaba 2018 cluster trace as per-node CPU/memory telemetry.

    Each call to get_reading() advances through the real 8-day trace for the
    specified node, applying per-node scaling so nodes have distinct steady-state
    utilisation while sharing the same temporal autocorrelation structure.

    Thread safety: not thread-safe (same as TelemetryCollector itself).
    """

    def __init__(self, csv_path: str | Path) -> None:
        """
        Load the Alibaba cluster trace CSV and build per-node lookup arrays.

        Args:
            csv_path: Path to ``alibaba_machine_usage_300s.csv``.
                      Expected columns: ``cpu_util_percent``, ``mem_util_percent``.

        Raises:
            FileNotFoundError: If csv_path does not exist.
            KeyError: If the CSV is missing expected column headers.
            ValueError: If the CSV contains no data rows.
        """
        cpu_vals: List[float] = []
        mem_vals: List[float] = []

        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cpu_vals.append(float(row["cpu_util_percent"]))
                mem_vals.append(float(row["mem_util_percent"]))

        if not cpu_vals:
            raise ValueError(f"No data rows found in {csv_path}")

        self._cpu_trace: np.ndarray = np.array(cpu_vals, dtype=np.float64)
        self._mem_trace: np.ndarray = np.array(mem_vals, dtype=np.float64)
        self._trace_len: int = len(cpu_vals)
        self._csv_path = Path(csv_path)

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_reading(self, node_id: str, tick_number: int) -> Tuple[float, float]:
        """
        Return (cpu_util_pct, mem_util_pct) for the given node at the given tick.

        The trace is treated as a circular buffer — once tick_number exceeds the
        8-day trace length, it wraps back to the start.

        Per-node offsets ensure that the 5 mock nodes observe different sections
        of the trace simultaneously, giving them independent temporal patterns.

        Args:
            node_id:     One of the 5 mock node IDs, or any unknown string.
            tick_number: Monotonically increasing integer (TelemetryCollector._tick_count).

        Returns:
            (cpu_util_pct, mem_util_pct) both clamped to [0.0, 100.0].
            Returns (_FALLBACK_CPU, _FALLBACK_MEM) for unknown node_ids.
        """
        config: Optional[Dict[str, float]] = _NODE_CONFIG.get(node_id)
        if config is None:
            return (_FALLBACK_CPU, _FALLBACK_MEM)

        # Circular index with per-node offset
        idx: int = (int(config["offset"]) + tick_number) % self._trace_len

        # Apply per-node transform to CPU
        raw_cpu: float = self._cpu_trace[idx]
        cpu = float(np.clip(raw_cpu * config["cpu_scale"] + config["cpu_bias"], 0.0, 100.0))

        # Memory: scale the high trace values into a more realistic range
        mem = float(np.clip(self._mem_trace[idx] * _MEM_SCALE, 0.0, 100.0))

        return cpu, mem

    @property
    def trace_length(self) -> int:
        """Number of 300-second timesteps in the loaded trace."""
        return self._trace_len

    def node_ids(self) -> List[str]:
        """The 5 mock node IDs that this adapter maps to trace segments."""
        return list(_NODE_CONFIG.keys())

    def __repr__(self) -> str:
        return (
            f"AlibabaMachineTraceAdapter("
            f"path={self._csv_path.name!r}, "
            f"trace_len={self._trace_len}, "
            f"nodes={len(_NODE_CONFIG)})"
        )


# ── BorgTraceAdapter ─────────────────────────────────────────────────────────
#
# Google Cluster Trace 2019 (Borg) adapter.
#
# The Borg CSV records individual task-level events with an `average_usage`
# field (a Python dict string) containing fractional CPU/memory values where
# 1.0 = 100% of one machine's capacity.
#
# Transformation pipeline:
#   1. Parse `average_usage` via regex (_BORG_USAGE_RE) → (cpu_frac, mem_frac)
#   2. Filter sentinel timestamps (Long.MAX_VALUE) and zero-usage rows
#   3. Sort by event time to preserve real temporal autocorrelation
#   4. Bucket into _N_BUCKETS time buckets (row-count-based, not wall-clock)
#   5. Normalise: P75 bucket → target baseline % (auto-computed at load time)
#   6. Apply per-node offset + scale/bias (same pattern as AlibabaMachineTraceAdapter)
#
# Compatible with the Kaggle "Google Cluster Trace 2019" sample CSV.
# Required columns: `time`, `average_usage`
#
# Per-node config targets (after P75 normalisation to ~40% mean):
#   node-cpu-01: ~35%  node-arm-02: ~20%  node-api-03: ~60%
#   node-gpu-04: ~45%  node-gpu-05: ~30%

_BORG_SENTINEL_TIME: int = 9_000_000_000_000_000_000  # filter Long.MAX_VALUE
_BORG_N_BUCKETS: int = 2000
_BORG_P75_TARGET_CPU: float = 40.0
_BORG_P75_TARGET_MEM: float = 50.0

_BORG_NODE_CONFIG: Dict[str, Dict[str, float]] = {
    "node-cpu-01": {"offset": 0,    "cpu_scale": 1.13, "cpu_bias":  0.0},
    "node-arm-02": {"offset": 400,  "cpu_scale": 0.65, "cpu_bias":  0.0},
    "node-api-03": {"offset": 800,  "cpu_scale": 1.00, "cpu_bias": 27.6},
    "node-gpu-04": {"offset": 1200, "cpu_scale": 1.00, "cpu_bias": 12.0},
    "node-gpu-05": {"offset": 1600, "cpu_scale": 0.97, "cpu_bias":  0.0},
}


class BorgTraceAdapter:
    """
    Replay Google Cluster Trace 2019 (Borg/GCE) data as per-node telemetry.

    Accepts the Kaggle "Google Cluster Trace 2019" CSV format — or any CSV
    with ``time`` (nanoseconds) and ``average_usage`` columns where
    average_usage is a dict string like ``{'cpus': 0.021, 'memory': 0.014}``.

    Usage:
        adapter = BorgTraceAdapter("borg_traces_data.csv")
        cpu_pct, mem_pct = adapter.get_reading("node-cpu-01", tick_number=42)

    The adapter builds a ~2000-bucket timeseries at load time. P75 of bucket
    averages is normalised to 40% CPU / 50% memory, giving a realistic range
    without clipping most values at 100%.
    """

    def __init__(self, csv_path: str | Path) -> None:
        """
        Load and pre-process the Borg trace CSV.

        Args:
            csv_path: Path to the Borg traces CSV.
                      Required columns: ``time``, ``average_usage``.

        Raises:
            FileNotFoundError: If csv_path does not exist.
            KeyError: If required columns are missing.
            ValueError: If no usable rows remain after filtering.
        """
        raw: List[Tuple[int, float, float]] = []  # (time_ns, cpu_frac, mem_frac)

        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    t = int(row["time"])
                    if t < 0 or t >= _BORG_SENTINEL_TIME:
                        continue
                    m_re = _BORG_USAGE_RE.search(row["average_usage"])
                    if m_re is None:
                        continue
                    cpu_s, mem_s = m_re.group(1), m_re.group(2)
                    if mem_s == "None":
                        continue
                    c, m = float(cpu_s), float(mem_s)
                    if c <= 0.0:
                        continue
                    raw.append((t, c, m))
                except Exception:
                    continue

        if not raw:
            raise ValueError(
                f"No usable rows found in {csv_path}. "
                "Ensure 'time' and 'average_usage' columns exist with valid data."
            )

        # Sort by event time to preserve temporal ordering
        raw.sort(key=lambda x: x[0])

        # Bucket into _N_BUCKETS row-count-based buckets and average each
        cpu_raw = np.array([r[1] for r in raw], dtype=np.float64)
        mem_raw = np.array([r[2] for r in raw], dtype=np.float64)

        chunk = max(1, len(raw) // _BORG_N_BUCKETS)
        cpu_buckets = np.array(
            [cpu_raw[i:i + chunk].mean() for i in range(0, len(raw), chunk)],
            dtype=np.float64,
        )
        mem_buckets = np.array(
            [mem_raw[i:i + chunk].mean() for i in range(0, len(raw), chunk)],
            dtype=np.float64,
        )

        # Auto-scale: P75 of buckets → target baseline %
        p75_cpu = float(np.percentile(cpu_buckets, 75))
        p75_mem = float(np.percentile(mem_buckets, 75))

        cpu_base_scale = (_BORG_P75_TARGET_CPU / (p75_cpu * 100.0)) if p75_cpu > 0 else 1.0
        mem_base_scale = (_BORG_P75_TARGET_MEM / (p75_mem * 100.0)) if p75_mem > 0 else 1.0

        self._cpu_trace: np.ndarray = np.clip(
            cpu_buckets * cpu_base_scale * 100.0, 0.0, 100.0
        )
        self._mem_trace: np.ndarray = np.clip(
            mem_buckets * mem_base_scale * 100.0, 0.0, 100.0
        )
        self._trace_len: int = len(self._cpu_trace)
        self._csv_path = Path(csv_path)
        self._n_raw_rows: int = len(raw)

    # ── Public API ───────────────────────────────────────────────────────────

    def get_reading(self, node_id: str, tick_number: int) -> Tuple[float, float]:
        """
        Return (cpu_util_pct, mem_util_pct) for the given node at the given tick.

        Same interface as AlibabaMachineTraceAdapter — drop-in compatible.

        Args:
            node_id:     One of the 5 mock node IDs, or any unknown string.
            tick_number: Monotonically increasing integer.

        Returns:
            (cpu_util_pct, mem_util_pct) both in [0.0, 100.0].
            Returns (40.0, 50.0) fallback for unknown node_ids.
        """
        config: Optional[Dict[str, float]] = _BORG_NODE_CONFIG.get(node_id)
        if config is None:
            return (40.0, 50.0)

        idx: int = (int(config["offset"]) + tick_number) % self._trace_len

        cpu = float(np.clip(
            self._cpu_trace[idx] * config["cpu_scale"] + config["cpu_bias"],
            0.0, 100.0,
        ))
        mem = float(np.clip(self._mem_trace[idx], 0.0, 100.0))
        return cpu, mem

    @property
    def trace_length(self) -> int:
        """Number of time buckets in the processed trace."""
        return self._trace_len

    @property
    def raw_rows(self) -> int:
        """Number of raw task events parsed from the CSV."""
        return self._n_raw_rows

    def node_ids(self) -> List[str]:
        """The 5 mock node IDs that this adapter maps to trace segments."""
        return list(_BORG_NODE_CONFIG.keys())

    def __repr__(self) -> str:
        return (
            f"BorgTraceAdapter("
            f"path={self._csv_path.name!r}, "
            f"raw_rows={self._n_raw_rows}, "
            f"buckets={self._trace_len})"
        )
