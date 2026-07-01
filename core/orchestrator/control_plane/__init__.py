"""
orchestrator/control_plane — the scheduling brain.

Public API (grows phase by phase):

    Phase 3:
        WorkloadPredictor  — LSTM-based CPU spike forecaster
                             reads WorkloadProfile → returns PredictionResult

    Phase 4:
        CostEngine         — composite node scoring for ACO η heuristic
                             score_node(job, node, prediction) → float [0,1]

    Phase 5:
        aco_schedule()          — replaces naive_schedule() in V1 scheduler.py
        naive_schedule()        — First Fit fallback (kept from V1)
        SchedulingFailedError   — raised when no node fits the job
        OrchestratorService     — V2 central control plane
        AdmissionRejectedError  — raised by admission control
        admit_job()             — admission check function

    Phase 6:
        SchedulingStrategy      — dataclass: node-filter + CostEngine overrides
        WorkloadIntentRouter    — maps JobRequest intent → SchedulingStrategy

    Phase 8:
        NodeAgent               — async per-node execution daemon
"""

from orchestrator.control_plane.predictor import WorkloadPredictor
from orchestrator.control_plane.cost_engine import CostEngine
from orchestrator.control_plane.scheduler import (
    aco_schedule,
    naive_schedule,
    SchedulingFailedError,
)
from orchestrator.control_plane.admission_controller import (
    AdmissionRejectedError,
    admit_job,
)
from orchestrator.control_plane.orchestration_service import OrchestratorService
from orchestrator.control_plane.intent_router import (
    SchedulingStrategy,
    WorkloadIntentRouter,
)
from orchestrator.telemetry.collector import TelemetryCollector
from orchestrator.data_plane.agent import NodeAgent

__all__ = [
    "WorkloadPredictor",
    "CostEngine",
    "aco_schedule",
    "naive_schedule",
    "SchedulingFailedError",
    "AdmissionRejectedError",
    "admit_job",
    "OrchestratorService",
    "SchedulingStrategy",
    "WorkloadIntentRouter",
    "TelemetryCollector",
    "NodeAgent",
]
