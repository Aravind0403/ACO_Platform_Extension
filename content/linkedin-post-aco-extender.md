# LinkedIn Post — ACO Extender (links to GitHub.io)

---

I've been building a compute scheduler using Ant Colony Optimisation — and the paper is under review at a peer-reviewed systems conference.

The core finding from the research: a 14-way ablation across LSTM, GRU, TCN, ARIMA, EMA, and others showed that EMA with α=0.5 beat LSTM on safe-node routing by 13.8 percentage points. The simplest predictor in the study won. The algorithm hit 0.82ms P99, 97.4% QoS compliance, 28.6% cost reduction over First-Fit — all validated on the Alibaba trace.

Then I tried to run it on an actual Kubernetes cluster.

Kubernetes doesn't care about ant colonies. It has its own scheduling logic — resource fit, node affinity, taints. If you want to change how it places pods, you have two options: replace the scheduler entirely, or use the extender protocol.

Most people don't know the extender protocol exists. It's an HTTP sidecar. After K8s runs its own filters, it forwards the candidate node list to your service, asks for scores, and uses them. No fork. No cluster-level changes.

I wired the ACO colony into that protocol.

The hard part wasn't the algorithm — that part was already validated. The hard parts were: PyTorch loading in 8 seconds on a sidecar that needs to respond in <1ms (fix: bypass the package __init__ entirely with importlib), Pydantic v2 dropping the __root__ pattern mid-build (fix: RootModel), and per-tenant pheromone isolation so a prod burst doesn't contaminate the research colony's trails.

The local demo runs in three terminals, no GKE needed. The Grafana dashboard shows pheromone convergence in real time — you can watch the colony lock onto the T4 node ($0.45/hr) over the V100 ($3.20/hr) after about 25 scheduling calls.

Full write-up on the extender architecture — protocol, scoring, import hack, what broke — is here: [link to GitHub.io post]

---

*Note: replace [link to GitHub.io post] with the actual URL once published.*
