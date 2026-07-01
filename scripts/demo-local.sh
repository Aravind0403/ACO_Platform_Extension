#!/usr/bin/env bash
# scripts/demo-local.sh
#
# One-command local demo — no GKE or minikube required.
#
# What it starts:
#   1. ACO extender     (FastAPI on :8080)
#   2. Prometheus        (Docker on :9090)
#   3. Grafana           (Docker on :3000)
#   4. Trace replay      (fires jobs at the extender)
#
# Usage:
#   chmod +x scripts/demo-local.sh
#   ./scripts/demo-local.sh
#
# Stop everything:
#   Ctrl+C — the script kills all background processes on exit.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Cleanup on exit ──────────────────────────────────────────────────────────
_pids=()
cleanup() {
    echo ""
    echo "==> Stopping all processes..."
    for pid in "${_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    echo "==> Stopping Docker services..."
    (cd observability && $DOCKER_COMPOSE down -v) 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── 1. Check dependencies ────────────────────────────────────────────────────
echo "==> Checking dependencies..."
command -v python3    >/dev/null || { echo "python3 not found"; exit 1; }
command -v uvicorn    >/dev/null || { echo "uvicorn not found — run: pip install uvicorn fastapi prometheus-client pydantic numpy"; exit 1; }
command -v docker     >/dev/null || { echo "docker not found"; exit 1; }
DOCKER_COMPOSE="docker compose"
command -v docker-compose >/dev/null 2>&1 && DOCKER_COMPOSE="docker-compose"
echo "    OK"

# ── 2. Start Prometheus + Grafana ────────────────────────────────────────────
echo "==> Starting Prometheus + Grafana..."
cd observability
$DOCKER_COMPOSE up -d
cd "$REPO_ROOT"
echo "    Prometheus: http://localhost:9090"
echo "    Grafana:    http://localhost:3000  (admin / admin)"

# ── 3. Start the ACO extender ────────────────────────────────────────────────
echo "==> Starting ACO extender on :8080..."
PYTHONPATH="$REPO_ROOT/core" \
  uvicorn extender.main:app --host 0.0.0.0 --port 8080 --log-level warning &
_pids+=($!)
echo "    Extender: http://localhost:8080"

# Wait for extender to be ready
echo "==> Waiting for extender to be ready..."
for i in $(seq 1 20); do
    if curl -sf http://localhost:8080/healthz >/dev/null 2>&1; then
        echo "    Ready."
        break
    fi
    sleep 1
done

# ── 4. Open Grafana in browser (optional) ────────────────────────────────────
if command -v open >/dev/null 2>&1; then
    sleep 2  # give Grafana a moment to load
    open "http://localhost:3000/d/aco-scheduler-v1" 2>/dev/null || true
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:3000/d/aco-scheduler-v1" 2>/dev/null || true
fi

# ── 5. Run trace replay ──────────────────────────────────────────────────────
echo ""
echo "==> Starting trace replay (firing jobs every 1s)..."
echo "    Watch pheromone converge in Grafana: http://localhost:3000"
echo "    Press Ctrl+C to stop."
echo ""
python3 scripts/trace_replay.py --interval 1.0 --verbose &
_pids+=($!)

# Keep the script alive
wait "${_pids[0]}"  # wait on extender — if it dies, cleanup triggers
