locals {
  name = var.name_prefix
}

# ---- image registry ------------------------------------------------------------------------

resource "aws_ecr_repository" "api" {
  name                 = "${local.name}-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ---- data and model artefacts --------------------------------------------------------------

resource "aws_s3_bucket" "artefacts" {
  bucket = var.artefact_bucket_name
}

resource "aws_s3_bucket_versioning" "artefacts" {
  bucket = aws_s3_bucket.artefacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artefacts" {
  bucket = aws_s3_bucket.artefacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artefacts" {
  bucket                  = aws_s3_bucket.artefacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_iam_policy_document" "artefacts_tls_only" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.artefacts.arn,
      "${aws_s3_bucket.artefacts.arn}/*",
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "artefacts" {
  bucket     = aws_s3_bucket.artefacts.id
  policy     = data.aws_iam_policy_document.artefacts_tls_only.json
  depends_on = [aws_s3_bucket_public_access_block.artefacts]
}

# ---- logs ----------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}-api"
  retention_in_days = var.log_retention_days
}

# ---- IAM: one role to start the task, one for what the running code may do -----------------

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "execution" {
  statement {
    sid       = "EcrAuthToken"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid       = "PullApiImage"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = [aws_ecr_repository.api.arn]
  }

  statement {
    sid       = "WriteLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.api.arn}:*"]
  }
}

resource "aws_iam_role_policy" "execution" {
  name   = "start-task"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution.json
}

resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "task" {
  statement {
    sid       = "ListArtefacts"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artefacts.arn]
  }

  statement {
    sid       = "ReadArtefacts"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.artefacts.arn}/*"]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "read-artefacts"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

# ---- network -------------------------------------------------------------------------------

resource "aws_security_group" "api" {
  name        = "${local.name}-api"
  description = "Risk API: port 8000 in from the allowed ranges, HTTPS out"
  vpc_id      = var.vpc_id

  ingress {
    description = "API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_ingress_cidrs
  }

  egress {
    description = "ECR, S3 and CloudWatch over HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---- service -------------------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = var.cpu_architecture
  }

  volume {
    name = "data"
  }

  volume {
    name = "mlruns"
  }

  volume {
    name = "tmp"
  }

  container_definitions = jsonencode([{
    name                   = "api"
    image                  = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
    essential              = true
    readonlyRootFilesystem = true
    user                   = "10001"

    portMappings = [{ containerPort = 8000, protocol = "tcp" }]

    environment = [
      { name = "RAIL_ARTEFACT_BUCKET", value = aws_s3_bucket.artefacts.bucket },
    ]

    mountPoints = [
      { sourceVolume = "data", containerPath = "/app/data", readOnly = false },
      { sourceVolume = "mlruns", containerPath = "/app/mlruns", readOnly = false },
      { sourceVolume = "tmp", containerPath = "/tmp", readOnly = false },
    ]

    healthCheck = {
      command = [
        "CMD-SHELL",
        "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)\"",
      ]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 90
    }

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name            = "${local.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = var.assign_public_ip
  }
}
