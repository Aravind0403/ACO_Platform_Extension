output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.aco.name
}

output "cluster_endpoint" {
  description = "GKE cluster API endpoint"
  value       = google_container_cluster.aco.endpoint
  sensitive   = true
}

output "region" {
  value = var.region
}

output "zone" {
  value = var.zone
}

output "kubeconfig_command" {
  description = "Run this after apply to configure kubectl"
  value       = "gcloud container clusters get-credentials ${var.cluster_name} --zone ${var.zone} --project ${var.project_id}"
}

output "node_pools" {
  description = "Summary of GPU node pools and their ACO cost labels"
  value = {
    cpu  = { machine = "n2-standard-4", cost_per_hour = "0.19", gpu = "none" }
    t4   = { machine = "n1-standard-4", cost_per_hour = "0.45", gpu = "t4" }
    a10  = { machine = "a2-highgpu-1g", cost_per_hour = "1.20", gpu = "a10" }
    p100 = { machine = "n1-standard-8", cost_per_hour = "1.60", gpu = "p100" }
    v100 = { machine = "n1-standard-8", cost_per_hour = "2.48", gpu = "v100" }
  }
}
