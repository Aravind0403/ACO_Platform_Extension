"""
orchestrator/telemetry — background telemetry ingestion and prediction pipeline.

Phase 7:   TelemetryCollector drives the prediction loop end-to-end.
Phase 7.5: AlibabaMachineTraceAdapter replaces synthetic noise with real trace data.
Phase 9:   BorgTraceAdapter adds Google Cluster Trace 2019 (Borg) support.

Public API:
    TelemetryCollector           — simulated node telemetry loop + LSTM refit trigger
    AlibabaMachineTraceAdapter   — Alibaba 2018 cluster trace replay adapter
    BorgTraceAdapter             — Google Cluster Trace 2019 (Borg/Kaggle) adapter
"""

from orchestrator.telemetry.collector import TelemetryCollector
from orchestrator.telemetry.trace_adapter import AlibabaMachineTraceAdapter, BorgTraceAdapter

__all__ = ["TelemetryCollector", "AlibabaMachineTraceAdapter", "BorgTraceAdapter"]
