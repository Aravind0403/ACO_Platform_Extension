# ACO Platform Extension

Platform extension for the [ACO Adaptive Compute Orchestrator](https://github.com/Aravind0403/ACO_Adaptive_Compute_Orchestrator) research repo.

Turns the scheduling algorithm (validated via trace simulation, HiPC 2026 submission) into a deployable Kubernetes system with full observability.

---

## What's in here

| Directory | What it does |
|---|---|
| `core/` | Core modules imported unchanged from the research repo (ACO engine, predictor, cost engine) |
| `extender/` | Kubernetes scheduler extender — FastAPI service K8s calls to get node bindings |
| `observability/` | Prometheus config + Grafana dashboard for routing %, cost/job, latency |
| `terraform/` | GKE infrastructure with GPU node pools per tier (V100, A10, T4, P100) |
| `k8s/` | Kubernetes manifests — scheduler config, namespaces, ResourceQuota |
| `scripts/` | Trace replay + load generation scripts |

---

## Build sequence

1. **Phase 1** — K8s scheduler extender (minikube local)
2. **Phase 2** — Prometheus + Grafana observability
3. **Phase 3** — Terraform GKE with GPU node groups
4. **Phase 4** — Multi-tenant isolation (namespace-per-tenant + ResourceQuota)

---

## Key numbers (from research repo)

- EMA α=0.5 → 89.2% routing to stable nodes vs LSTM 75.4% (CPU topology)
- GPU confirmatory → EMA 100% vs LSTM 33.1% — 7 GPU types, $0.45–$3.20/GPU-hr
- P99 scheduling latency: <1ms up to 200 concurrent jobs
- Datasets: Alibaba 2018 Machine Trace + Alibaba OpenB GPU Trace (ATC'23)

---

## Requirements

- Python 3.11+
- Docker
- minikube (Phase 1)
- Terraform ≥ 1.6 (Phase 3)
- kubectl
