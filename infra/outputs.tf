output "ecr_repository_url" {
  description = "Push the api image here."
  value       = aws_ecr_repository.api.repository_url
}

output "artefact_bucket" {
  description = "Bucket for data and model artefacts."
  value       = aws_s3_bucket.artefacts.bucket
}

output "ecs_service" {
  value = aws_ecs_service.api.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.api.name
}
