terraform {
  required_version = ">= 1.9.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "= 5.100.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # With offline_plan set, the provider makes no call to AWS at plan time, so `terraform plan`
  # works with no account. Leave it false for a real plan or apply.
  skip_credentials_validation = var.offline_plan
  skip_requesting_account_id  = var.offline_plan
  skip_metadata_api_check     = var.offline_plan
  skip_region_validation      = var.offline_plan

  default_tags {
    tags = {
      project   = "connection-risk-engine"
      managedBy = "terraform"
    }
  }
}
