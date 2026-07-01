# ACO Platform Extension — Demo Walkthrough

Two ways to run this: **local mode** (no cloud needed, good for interviews and HiPC) and **GKE mode** (full cloud deployment). Start with local.

---

## Local Mode (recommended for demo)

### Prerequisites

```bash
pip install fastapi uvicorn prometheus-client pydantic numpy
docker  # for Prometheus + Grafana
```

### Step 1 — Install Python dependencies

```bash
pip install fastapi "uvicorn[standard]" prometheus-client pydantic numpy
```

### Step 2 — Start everything with one command

From the repo root:

```bash
chmod +x scripts/demo-local.sh
./scripts/demo-local.sh
```

This starts (in order):
- Prometheus on http://localhost:9090
- Grafana on http://localhost:3000
- ACO extender on http://localhost:8080
- Trace replay firing jobs every 1 second

Grafana opens automatically. Log in with **admin / admin**.

### Step 3 — Open the dashboard

Go to **Dashboards → ACO Scheduler — Platform Extension**.

The interesting panels to watch:
- **Pheromone Level per Node** — watch T4 (cheapest GPU) pull ahead as ACO converges
- **Node Selection Rate** — routing % per node, should stabilise toward T4 for GPU jobs
- **Cost per Job** — median $/hr should trend down as pheromone converges
- **P99 Latency** — should stay well under 10ms

### Step 4 — Run a focused GPU workload (optional)

Open a second terminal and fire GPU-only jobs to accelerate convergence:

```bash
python3 scripts/trace_replay.py --interval 0.5 --verbose
```

### Step 5 — Reset between demo runs

When you want to show the convergence story again from scratch:

```bash
# Reset all pheromone state (extender forgets everything it learned)
curl -X POST http://localhost:8080/reset

# Or reset just one tenant
curl -X POST "http://localhost:8080/reset?tenant=research"
```

Grafana history is **not** affected — Prometheus keeps its TSDB. So you get a clean before/after in the same dashboard: flat pheromone → reset → convergence again. Good for live demos.

### Step 6 — Stop

`Ctrl+C` in the demo-local.sh terminal. It kills all background processes and tears down Docker.

---

## GKE Mode (cloud deployment)

### Prerequisites

- [Terraform ≥ 1.6](https://developer.hashicorp.com/terraform/install)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- GCP project with billing enabled

### Step 1 — Enable APIs

```bash
gcloud services enable container.googleapis.com compute.googleapis.com
```

### Step 2 — Provision the cluster

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — set your project_id

terraform init
terraform plan   # review what will be created
terraform apply  # takes ~10 minutes
```

### Step 3 — Configure kubectl

The apply output prints the exact command. Run it:

```bash
gcloud container clusters get-credentials aco-platform \
  --zone us-central1-a --project <your-project>
```

Verify nodes are up:

```bash
kubectl get nodes --show-labels | grep aco/gpu-type
```

You should see nodes labelled `t4`, `a10`, `p100`, `v100`.

### Step 4 — Deploy the extender

```bash
# Build and push the image (replace PROJECT_ID)
docker build -f extender/Dockerfile -t gcr.io/PROJECT_ID/aco-extender:latest .
docker push gcr.io/PROJECT_ID/aco-extender:latest

# Update image reference in the manifest, then deploy
kubectl apply -f k8s/extender-deployment.yaml
kubectl rollout status deployment/aco-extender -n aco-system
```

### Step 5 — Apply tenant namespaces + quotas

```bash
kubectl apply -f k8s/tenants.yaml
kubectl get resourcequota -A
```

### Step 6 — Wire in the custom scheduler

```bash
kubectl apply -f k8s/scheduler-config.yaml
```

### Step 7 — Deploy Prometheus + Grafana (in-cluster)

Switch the Prometheus target in `observability/prometheus.yml` from `host.docker.internal:8080` to the in-cluster service:

```yaml
- targets:
    - "aco-extender.aco-system.svc.cluster.local:8080"
```

Then deploy with Helm (quickest):

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# Port-forward Grafana locally
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
```

Import `observability/grafana-dashboard.json` via Grafana UI → Dashboards → Import.

### Step 8 — Reset between demo runs (GKE)

```bash
# Port-forward the extender locally first
kubectl port-forward svc/aco-extender 8080:8080 -n aco-system

# Then reset
curl -X POST http://localhost:8080/reset
```

### Tear down (important — GPU nodes are expensive)

```bash
cd terraform
terraform destroy
```

Always destroy after demoing. Default config costs ~$6/hr.

---

## What to show in the demo (45-second story)

1. **Start fresh** — pheromone is flat across all nodes. ACO has no preference.
2. **Fire GPU batch jobs** — first few jobs spread randomly (exploration phase).
3. **Watch T4 emerge** — cost_efficiency_factor is highest on T4 ($0.45/hr). Pheromone deposits faster there. Within ~20 jobs the T4 bar is visibly ahead.
4. **Show V100 being avoided** — it's scoring low for batch jobs despite being available. The algorithm is doing cost-aware routing without being told to avoid V100s explicitly.
5. **Reset and repeat** — `curl -X POST http://localhost:8080/reset`. Pheromone goes flat again. Convergence happens again. Reproducible.
6. **Switch to latency-critical jobs** — CPU nodes score higher (spot reliability factor kicks in). Watch routing shift.

---

## Troubleshooting

**Extender won't start**
```bash
PYTHONPATH=core python3 -c "from extender.main import app; print('OK')"
```

**Prometheus not scraping**
- Check `http://localhost:9090/targets` — extender should show as UP
- On macOS, `host.docker.internal` resolves automatically. On Linux, add `--add-host=host.docker.internal:host-gateway` to the Prometheus docker run command.

**Grafana dashboard missing**
- Go to Dashboards → Import → upload `observability/grafana-dashboard.json`
- Or browse to `http://localhost:3000/d/aco-scheduler-v1` directly

**No data in panels**
- Confirm the replay script is running: you should see job lines printing in the terminal
- Check `http://localhost:8080/healthz` — `tenants` should be non-empty after a few jobs
