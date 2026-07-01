#!/usr/bin/env bash
# scripts/minikube-start.sh
#
# Start minikube, build the extender image inside it, and deploy everything.
# Run from the repo root:
#   chmod +x scripts/minikube-start.sh
#   ./scripts/minikube-start.sh

set -euo pipefail

echo "==> Starting minikube..."
minikube start --cpus=4 --memory=6g

echo "==> Pointing Docker to minikube's daemon (so the image lands inside minikube)..."
eval "$(minikube docker-env)"

echo "==> Building the extender image..."
docker build -f extender/Dockerfile -t aco-extender:latest .

echo "==> Deploying namespace + extender..."
kubectl apply -f k8s/extender-deployment.yaml

echo "==> Waiting for extender pod to be ready..."
kubectl rollout status deployment/aco-extender -n aco-system --timeout=120s

echo ""
echo "==> Done. Extender is running. Test it:"
echo "    kubectl port-forward svc/aco-extender 8080:8080 -n aco-system"
echo "    curl http://localhost:8080/healthz"
echo ""
echo "==> To wire in the custom scheduler config, restart the scheduler with:"
echo "    minikube ssh 'sudo cp /dev/stdin /etc/kubernetes/scheduler-config.yaml' < k8s/scheduler-config.yaml"
