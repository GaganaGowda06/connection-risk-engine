# Anything that depends on an account has no default, so it has to be chosen on purpose.

variable "aws_region" {
  description = "Region to deploy to."
  type        = string
}

variable "vpc_id" {
  description = "VPC the service runs in."
  type        = string
}

variable "subnet_ids" {
  description = "Subnets for the Fargate tasks. Private subnets with a NAT, or public subnets with assign_public_ip."
  type        = list(string)
}

variable "allowed_ingress_cidrs" {
  description = "CIDR blocks that may reach port 8000. Put a load balancer or an internal range here, not 0.0.0.0/0."
  type        = list(string)
}

variable "artefact_bucket_name" {
  description = "Globally unique name for the S3 bucket that holds data and model artefacts."
  type        = string
}

variable "image_tag" {
  description = "Tag of the api image in the ECR repository, for example a git commit sha."
  type        = string
}

variable "assign_public_ip" {
  description = "Give each task a public IP. Needed only when the subnets have no NAT."
  type        = bool
  default     = false
}

variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
  default     = "connection-risk"
}

variable "cpu_architecture" {
  description = "Architecture the image was built for."
  type        = string
  default     = "X86_64"

  validation {
    condition     = contains(["X86_64", "ARM64"], var.cpu_architecture)
    error_message = "cpu_architecture must be X86_64 or ARM64."
  }
}

variable "task_cpu" {
  description = "Fargate CPU units. 1024 is one vCPU."
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "Fargate memory in MiB. 2048 is 2 GB."
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Number of tasks."
  type        = number
  default     = 1
}

variable "log_retention_days" {
  description = "CloudWatch log retention."
  type        = number
  default     = 14
}

variable "offline_plan" {
  description = "Skip every call to AWS at plan time, so a plan runs with no account."
  type        = bool
  default     = false
}
