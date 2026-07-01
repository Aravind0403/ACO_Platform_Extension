# Terraform Deploy Guide

Provisions a GKE cluster with 5 node pools (CPU + 4 GPU tiers) for the ACO scheduler.

## Prerequisites

- [Terraform ≥ 1.6](https://developer.hashicorp.com/terraform/install)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) — authenticated with `gcloud auth login`
- GCP project with the following APIs enabled:
  ```
  gcloud services enable container.googleapis.com compute.googleapis.com
  ```

## Steps

**1. Configure your variables**
```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — set project_id at minimum
```

**2. Initialise Terraform**
```bash
terraform init
```

**3. Preview what will be created**
```bash
terraform plan
```

**4. Apply**
```bash
terraform apply
```
Takes ~10 minutes. Creates the cluster + all 5 node pools.

**5. Wire up kubectl**

The exact command is printed in the `apply` output under `kubeconfig_command`. It looks like:
```bash
gcloud container clusters get-credentials aco-platform --zone us-central1-a --project <your-project>
```

**6. Verify nodes are ready**
```bash
kubectl get nodes --show-labels
```
You should see nodes with `aco/gpu-type=t4`, `aco/gpu-type=v100`, etc.

## Tear down

```bash
terraform destroy
```

GPU nodes are expensive — always destroy when not actively demoing.

## Cost estimate (us-central1, default config)

| Pool   | Machine        | GPU  | $/hr   |
|--------|----------------|------|--------|
| cpu ×2 | n2-standard-4  | —    | ~$0.38 |
| t4 ×1  | n1-standard-4  | T4   | ~$0.45 |
| a10 ×1 | a2-highgpu-1g  | A10  | ~$1.20 |
| p100 ×1| n1-standard-8  | P100 | ~$1.60 |
| v100 ×1| n1-standard-8  | V100 | ~$2.48 |
| **Total** | | | **~$6.11/hr** |

Set `*_node_count = 0` in `terraform.tfvars` for tiers you don't need to save cost.
