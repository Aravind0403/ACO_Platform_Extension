"""
orchestrator/telemetry/collector.py
─────────────────────────────────────
TelemetryCollector: drives the live telemetry → prediction → cache pipeline.

What this is
─────────────
Phase 7 closes the gap between Phases 3–6:

  Before Phase 7:
    - WorkloadPredictor exists but is never trained in production (no data source)
    - CostEngine.sla_headroom_factor() always falls back to allocation estimates
      because node.latest_telemetry is always None
    - _prediction_cache in OrchestratorService is always empty

  After Phase 7:
    - TelemetryCollector.tick() is called periodically (simulated or real)
    - Each tick generates NodeTelemetry for every node in the cluster
    - Telemetry is ingested via OrchestratorService.update_node_telemetry()
      → node.latest_telemetry is populated → CostEngine gets real CPU util
    - A per-node WorkloadProfile accumulates ResourceSamples from each tick
    - Every REFIT_INTERVAL ticks: predictors are refitted and cache updated
      → aco_schedule() starts receiving spike predictions

V2 vs V3
──────────
V2 (this file): simulated telemetry. Each node has a configurable baseline
CPU utilisation. Random Gaussian noise is added to simulate realistic variance.
GPU nodes get synthetic GPU utilisation. No actual network calls.

V3: replace _generate_telemetry() with a real NodeAgent HTTP heartbeat.
The rest of the pipeline (profile accumulation, refit, cache) stays identical.

Design
───────
- Synchronous (no asyncio). tick() is cheap enough to call from any context.
  Phase 9 will wrap it in asyncio.create_task() with an interval.
- Per-node profiles — not shared across nodes or workload types.
  Each predictor is trained on its own node's CPU history.
  This is the Phase 7 fix to the Phase 5 approximation where all predictors
  shared the BATCH workload profile.
- inject_spike() supports test scenarios: temporarily raises a node's baseline
  to simulate a CPU spike (92% — above the 90% spike detection threshold in
  predictor.py), verifying that the prediction cache correctly warns aco_schedule.

Integration contract
─────────────────────
    collector = TelemetryCollector(orchestration_service)
    collector.tick()                    # call periodically
    collector.inject_spike("node-cpu-01", n_ticks=5)  # for tests

All effects are visible through orchestration_service:
  - orchestration_service.node_state[node_id].latest_telemetry is updated
  - orchestration_service._prediction_cache[node_id] is refreshed after refit
  - orchestration_service.predictors[node_id] is refitted on its own data
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, Optional

from orchestrator.shared.models import NodeArch, NodeTelemetry, PredictionResult
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile

if TYPE_CHECKING:
    from orchestrator.telemetry.trace_adapter import AlibabaMachineTraceAdapter

# ── Constants ─────────────────────────────────────────────────────────────────

CPU_NOISE_STD: float = 5.0
"""Gaussian noise standard deviation added to each node's baseline CPU util.

5% std gives realistic short-term variance without generating false spikes.
A ±2σ excursion (10%) will occasionally push a 35%-baseline node to 45%,
but will not reach the 90% spike threshold — keeping false positives low.
"""

MEMORY_BASE_UTIL: float = 50.0
"""Baseline memory utilisation applied uniformly across all nodes (%).

Memory is typically more stable than CPU — constant at 50% ± MEM_NOISE_STD.
"""

MEMORY_NOISE_STD: float = 10.0
"""Gaussian noise std dev for memory utilisation (%)."""

GPU_BASE_UTIL: float = 70.0
"""Baseline GPU utilisation for GPU nodes (%) — represents active inference/training."""

GPU_NOISE_STD: float = 15.0
"""Gaussian noise std dev for GPU utilisation (%)."""

REFIT_INTERVAL: int = 10
"""Number of tick() calls between predictor refits.

Refitting the LSTM (50 epochs) takes ~50ms. Refitting every tick would add
50ms of latency to the scheduling hot path for no benefit. Every 10 ticks
gives fresh predictions at 1/10th the cost.

At 1 tick/second (typical in V3): refit every 10 seconds. At 1 tick/30s
(typical in tests): refit every 300 simulated seconds. Either cadence is
sufficient for the LSTM's 5-minute forecast horizon.
"""

SPIKE_CPU_UTIL: float = 92.0
"""CPU utilisation injected by inject_spike().

Must be above the 90% threshold used in predictor.py's spike probability
calculation. At 92%, spike_probability will rise significantly, allowing
aco_schedule() to prefer non-spiking nodes.
"""

# Default baseline CPU utilisations per mock node.
# These mirror the mock cluster in OrchestratorService._initialize_mock_nodes().
_DEFAULT_BASELINES: Dict[str, float] = {
    "node-cpu-01": 35.0,   # general-purpose batch: moderate steady load
    "node-arm-02": 20.0,   # cheap spot ARM64: often idle between batch jobs
    "node-api-03": 60.0,   # API gateway: higher steady-state from constant requests
    "node-gpu-04": 45.0,   # GPU on-demand: moderate (spikes during training)
    "node-gpu-05": 30.0,   # GPU spot: lower baseline (less frequently scheduled)
}

# GPU node arch values — used to decide whether to emit gpu_util_pct
_GPU_ARCH = NodeArch.GPU_NODE


class TelemetryCollector:
    """
    Simulated telemetry ingestion loop for V2.

    Drives the OrchestratorService prediction pipeline by:
      1. Generating synthetic NodeTelemetry per node (Gaussian noise around baseline)
      2. Calling orchestration_service.update_node_telemetry() for each node
      3. Accumulating ResourceSamples in per-node WorkloadProfiles
      4. Every REFIT_INTERVAL ticks: refitting per-node LSTM predictors
         and refreshing the _prediction_cache

    V3 upgrade path:
      - Replace _generate_telemetry() with a real HTTP heartbeat call
      - Keep tick(), _refit(), inject_spike() unchanged
      - The rest of the pipeline is identical

    Usage:
        svc = OrchestratorService()
        collector = TelemetryCollector(svc)
        for _ in range(100):
            collector.tick()   # call periodically (every 1s in V3)
    """

    def __init__(
        self,
        orchestration_service: "OrchestratorService",  # type: ignore[name-defined]
        trace_adapter: Optional["AlibabaMachineTraceAdapter"] = None,
    ) -> None:
        """
        Initialise the collector bound to an OrchestratorService instance.

        Args:
            orchestration_service: The live OrchestratorService. The collector
                                   holds a reference and mutates its state
                                   via update_node_telemetry() and _prediction_cache.
            trace_adapter:         Optional AlibabaMachineTraceAdapter. When provided,
                                   _generate_telemetry() replays real Alibaba cluster
                                   trace data instead of random.gauss() noise.
                                   inject_spike() continues to work in both modes.
                                   Defaults to None (keeps original Gaussian path).
        """
        self._service = orchestration_service
        self._trace_adapter = trace_adapter
        self._tick_count: int = 0

        # Per-node baseline CPU utilisations (mutable — inject_spike() overrides these)
        self._base_cpu: Dict[str, float] = {}
        self._spike_remaining: Dict[str, int] = {}   # node_id → ticks remaining
        self._pre_spike_base: Dict[str, float] = {}  # original base before spike

        # Populate baselines from known nodes, falling back to _DEFAULT_BASELINES
        for node_id in self._service.node_state:
            self._base_cpu[node_id] = _DEFAULT_BASELINES.get(node_id, 40.0)

        # Per-node WorkloadProfiles — accumulate CPU history for predictor training
        # Keyed by node_id, not by workload type (unlike the global workload_profiles).
        # This gives each node an independent training signal.
        self._per_node_profiles: Dict[str, WorkloadProfile] = {
            node_id: WorkloadProfile(workload_name=f"node:{node_id}")
            for node_id in self._service.node_state
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """
        Execute one telemetry collection cycle.

        For each node in the cluster:
          1. Generate a NodeTelemetry snapshot (synthetic in V2)
          2. Feed it to OrchestratorService.update_node_telemetry()
             → updates node.latest_telemetry (CostEngine reads this)
          3. Add a ResourceSample to this node's per-node profile

        Every REFIT_INTERVAL ticks, also:
          4. Refit all per-node predictors and refresh _prediction_cache
        """
        self._tick_count += 1

        for node_id in list(self._service.node_state.keys()):
            # Step 1: generate telemetry
            telemetry = self._generate_telemetry(node_id)

            # Step 2: push to service (updates node.latest_telemetry)
            self._service.update_node_telemetry(telemetry)

            # Step 3: accumulate in per-node profile
            sample = ResourceSample(
                cpu_cores_used=telemetry.cpu_util_pct * 0.01 * self._service.node_state[node_id].total_cpu_cores,
                memory_gb_used=telemetry.memory_util_pct * 0.01 * self._service.node_state[node_id].total_memory_gb,
                duration_s=1.0,           # each tick represents ~1s of observation
                scheduling_latency_ms=0.0,
            )
            if node_id in self._per_node_profiles:
                self._per_node_profiles[node_id].add_sample(sample)

            # Decrement spike counter if active
            if node_id in self._spike_remaining:
                self._spike_remaining[node_id] -= 1
                if self._spike_remaining[node_id] <= 0:
                    # Restore original baseline
                    self._base_cpu[node_id] = self._pre_spike_base.pop(node_id, self._base_cpu[node_id])
                    del self._spike_remaining[node_id]

        # Step 4: periodic refit
        if self._tick_count % REFIT_INTERVAL == 0:
            self._refit()

    def inject_spike(self, node_id: str, n_ticks: int = 5) -> None:
        """
        Temporarily raise a node's baseline CPU to SPIKE_CPU_UTIL for n_ticks ticks.

        Used in tests to verify:
          - NodeTelemetry reflects the spike (cpu_util_pct ≈ 92%)
          - Per-node profile captures the spike samples
          - After refit: spike_probability in _prediction_cache rises
          - aco_schedule prefers the non-spiking node

        The spike automatically resets after n_ticks. Calling inject_spike()
        again before the counter expires extends the spike by an additional n_ticks.

        Args:
            node_id: Which node to spike. Unknown node_id is silently ignored.
            n_ticks: How many subsequent tick() calls to spike for. Default 5.
        """
        if node_id not in self._base_cpu:
            return   # unknown node — silently ignore, same as update_node_telemetry

        # Save the pre-spike baseline (only on first injection, not on extension)
        if node_id not in self._spike_remaining:
            self._pre_spike_base[node_id] = self._base_cpu[node_id]

        self._base_cpu[node_id] = SPIKE_CPU_UTIL
        self._spike_remaining[node_id] = n_ticks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _generate_telemetry(self, node_id: str) -> NodeTelemetry:
        """
        Generate a NodeTelemetry snapshot for one node.

        Two paths depending on whether a trace_adapter was provided:

        Trace-replay path (trace_adapter is not None):
          cpu_util, mem_util = adapter.get_reading(node_id, tick_count)
          If a spike is active for this node, cpu_util is overridden to SPIKE_CPU_UTIL.
          Memory is taken from the trace (pre-scaled by the adapter to ~39–48%).
          GPU utilisation is still synthetic (Gaussian) — the Alibaba trace has no GPU signal.

        Gaussian path (trace_adapter is None — default):
          cpu_util_pct    = clamp(N(base_cpu, CPU_NOISE_STD), 0, 100)
          memory_util_pct = clamp(N(MEMORY_BASE_UTIL, MEMORY_NOISE_STD), 0, 100)
          Spike is built into base_cpu (inject_spike sets _base_cpu[node_id] = 92%).

        Args:
            node_id: The node to generate telemetry for.

        Returns:
            NodeTelemetry snapshot for this tick.
        """
        node = self._service.node_state.get(node_id)

        if self._trace_adapter is not None:
            # Real trace path: get autocorrelated CPU/memory from Alibaba trace
            cpu_util, mem_util = self._trace_adapter.get_reading(node_id, self._tick_count)
            # Spike override: if a spike is active, replace the trace CPU with spike level
            if node_id in self._spike_remaining:
                cpu_util = SPIKE_CPU_UTIL
        else:
            # Gaussian path: original behaviour (inject_spike modifies _base_cpu directly)
            cpu_base = self._base_cpu.get(node_id, 40.0)
            cpu_util = _clamp(random.gauss(cpu_base, CPU_NOISE_STD), 0.0, 100.0)
            mem_util = _clamp(random.gauss(MEMORY_BASE_UTIL, MEMORY_NOISE_STD), 0.0, 100.0)

        # GPU nodes get synthetic GPU utilisation; CPU/ARM nodes don't
        # (Alibaba trace has no GPU data — always use Gaussian for GPU nodes)
        gpu_util: Dict[str, float] = {}
        if node is not None and node.arch == _GPU_ARCH:
            gpu_util = {
                "A100": _clamp(random.gauss(GPU_BASE_UTIL, GPU_NOISE_STD), 0.0, 100.0)
            }

        return NodeTelemetry(
            node_id=node_id,
            cpu_util_pct=cpu_util,
            memory_util_pct=mem_util,
            gpu_util_pct=gpu_util,
        )

    def _refit(self) -> None:
        """
        Refit all per-node predictors and refresh the prediction cache.

        Called every REFIT_INTERVAL ticks. For each node:
          1. Call predictor.refit_if_needed(per_node_profile) — only refits
             if ≥10 new samples accumulated since last fit (REFIT_THRESHOLD)
          2. Call predictor.predict(per_node_profile) — cold-start safe
          3. Store result in service._prediction_cache[node_id]

        This is the Phase 7 fix to Phase 5's approximation:
          Phase 5: all node predictors used the global BATCH WorkloadProfile
          Phase 7: each node predictor uses its own per-node profile
                   → independent training on each node's actual CPU history
        """
        for node_id, profile in self._per_node_profiles.items():
            if node_id not in self._service.predictors:
                continue

            predictor = self._service.predictors[node_id]

            # Only refit if we have at least 1 sample (predict() needs has_enough_data
            # to be True for the trained path; cold-start handles the rest)
            if len(profile.samples) == 0:
                continue

            predictor.refit_if_needed(profile)

            result = predictor.predict(profile)

            # Store in service cache with the correct node_id
            # (predictor.predict() returns PredictionResult with its own node_id which
            # may be the profile workload_name — we override it with the real node_id)
            self._service._prediction_cache[node_id] = PredictionResult(
                node_id=node_id,
                forecast_horizon_min=result.forecast_horizon_min,
                predicted_cpu_util=result.predicted_cpu_util,
                predicted_memory_util=result.predicted_memory_util,
                predicted_gpu_util=result.predicted_gpu_util,
                spike_probability=result.spike_probability,
                confidence=result.confidence,
            )

    # ── Introspection ──────────────────────────────────────────────────────────

    def get_per_node_profile(self, node_id: str) -> Optional[WorkloadProfile]:
        """Return the per-node WorkloadProfile for a given node_id. None if unknown."""
        return self._per_node_profiles.get(node_id)

    @property
    def tick_count(self) -> int:
        """Total number of tick() calls since this collector was created."""
        return self._tick_count

    def __repr__(self) -> str:
        return (
            f"TelemetryCollector("
            f"nodes={len(self._per_node_profiles)}, "
            f"ticks={self._tick_count}, "
            f"refit_interval={REFIT_INTERVAL})"
        )


# ── Utility ────────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]. Equivalent to max(lo, min(hi, value))."""
    return max(lo, min(hi, value))
