# General Configuration
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "enterprise-ai-gateway"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "devops-team"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "engineering"
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the Amazon EKS public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Node Group Configuration
variable "node_groups" {
  description = "EKS node group configurations"
  type = object({
    general = object({
      instance_types = list(string)
      min_size       = number
      max_size       = number
      desired_size   = number
    })
    compute = object({
      instance_types = list(string)
      min_size       = number
      max_size       = number
      desired_size   = number
    })
    memory = object({
      instance_types = list(string)
      min_size       = number
      max_size       = number
      desired_size   = number
    })
  })
  default = {
    general = {
      instance_types = ["m5.large", "m5.xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }
    compute = {
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      min_size       = 0
      max_size       = 10
      desired_size   = 2
    }
    memory = {
      instance_types = ["r5.2xlarge", "r5.4xlarge"]
      min_size       = 0
      max_size       = 8
      desired_size   = 1
    }
  }
}

# Authentication Configuration
variable "aws_auth_roles" {
  description = "List of role maps to add to the aws-auth configmap"
  type = list(object({
    rolearn  = string
    username = string
    groups   = list(string)
  }))
  default = []
}

variable "aws_auth_users" {
  description = "List of user maps to add to the aws-auth configmap"
  type = list(object({
    userarn  = string
    username = string
    groups   = list(string)
  }))
  default = []
}

variable "aws_auth_accounts" {
  description = "List of account maps to add to the aws-auth configmap"
  type        = list(string)
  default     = []
}

# Database Configuration
variable "rds_configuration" {
  description = "RDS configuration for PostgreSQL"
  type = object({
    instance_class    = string
    allocated_storage = number
    engine_version    = string
    backup_retention  = number
    multi_az         = bool
    deletion_protection = bool
  })
  default = {
    instance_class    = "db.t3.medium"
    allocated_storage = 100
    engine_version    = "15.4"
    backup_retention  = 7
    multi_az         = true
    deletion_protection = true
  }
}

# Redis Configuration
variable "redis_configuration" {
  description = "ElastiCache Redis configuration"
  type = object({
    node_type           = string
    num_cache_nodes     = number
    parameter_group     = string
    engine_version      = string
    backup_retention    = number
    automatic_failover  = bool
  })
  default = {
    node_type           = "cache.r6g.large"
    num_cache_nodes     = 3
    parameter_group     = "default.redis7"
    engine_version      = "7.0"
    backup_retention    = 5
    automatic_failover  = true
  }
}

# Monitoring Configuration
variable "monitoring_configuration" {
  description = "Monitoring and observability configuration"
  type = object({
    enable_cloudwatch_logs     = bool
    log_retention_days        = number
    enable_xray_tracing       = bool
    enable_container_insights = bool
    enable_prometheus         = bool
    enable_grafana           = bool
    enable_jaeger            = bool
  })
  default = {
    enable_cloudwatch_logs     = true
    log_retention_days        = 30
    enable_xray_tracing       = true
    enable_container_insights = true
    enable_prometheus         = true
    enable_grafana           = true
    enable_jaeger            = true
  }
}

# Security Configuration
variable "security_configuration" {
  description = "Security configuration"
  type = object({
    enable_secrets_manager = bool
    enable_parameter_store = bool
    enable_waf             = bool
    enable_shield         = bool
    enable_guardduty      = bool
  })
  default = {
    enable_secrets_manager = true
    enable_parameter_store = true
    enable_waf             = true
    enable_shield         = false
    enable_guardduty      = true
  }
}

# Backup Configuration
variable "backup_configuration" {
  description = "Backup configuration"
  type = object({
    enable_backup_vault = bool
    backup_schedule    = string
    retention_days     = number
    cross_region_backup = bool
  })
  default = {
    enable_backup_vault = true
    backup_schedule    = "cron(0 2 ? * * *)"  # Daily at 2 AM
    retention_days     = 30
    cross_region_backup = true
  }
}

# Scaling Configuration
variable "scaling_configuration" {
  description = "Auto-scaling configuration"
  type = object({
    enable_cluster_autoscaler = bool
    enable_vpa               = bool
    enable_hpa               = bool
    metrics_server_enabled   = bool
  })
  default = {
    enable_cluster_autoscaler = true
    enable_vpa               = true
    enable_hpa               = true
    metrics_server_enabled   = true
  }
}