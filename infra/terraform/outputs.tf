output "job_name" {
  description = "AI Training job name. Inspect with: ovhai job get <job_name>"
  value       = local.job_name
}

output "follow_logs_command" {
  description = "Copy-paste to stream the job logs once it starts."
  value       = "ovhai job logs ${local.job_name} --follow"
}

output "status_command" {
  description = "Copy-paste to check the job status."
  value       = "ovhai job get ${local.job_name}"
}

output "cancel_command" {
  description = "Copy-paste to cancel the job. terraform destroy does NOT call this."
  value       = "ovhai job kill ${local.job_name}"
}
