terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket         = "enterprise-ai-gateway-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "enterprise-ai-gateway-terraform-locks"
    encrypt        = true
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment   = var.environment
      Project       = "enterprise-ai-gateway"
      ManagedBy     = "terraform"
      Owner         = var.owner
      CostCenter    = var.cost_center
      Backup        = "required"
      Monitoring    = "enabled"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Enable flow logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"

  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true

  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs

  # Encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # IRSA
  enable_irsa = true

  # Node groups
  eks_managed_node_groups = {
    general = {
      name = "general"

      instance_types = var.node_groups.general.instance_types
      ami_type       = "AL2_x86_64"
      
      min_size     = var.node_groups.general.min_size
      max_size     = var.node_groups.general.max_size
      desired_size = var.node_groups.general.desired_size

      disk_size = 50
      disk_type = "gp3"
      
      labels = {
        Environment = var.environment
        NodeGroup   = "general"
      }

      taints = []

      update_config = {
        max_unavailable_percentage = 25
      }

      # Launch template
      create_launch_template = false
      launch_template_name   = ""

      remote_access = {
        ec2_ssh_key = aws_key_pair.eks_nodes.key_name
      }
    }

    compute = {
      name = "compute-intensive"

      instance_types = var.node_groups.compute.instance_types
      ami_type       = "AL2_x86_64"
      
      min_size     = var.node_groups.compute.min_size
      max_size     = var.node_groups.compute.max_size
      desired_size = var.node_groups.compute.desired_size

      disk_size = 100
      disk_type = "gp3"

      labels = {
        Environment = var.environment
        NodeGroup   = "compute-intensive"
        workload    = "ai-processing"
      }

      taints = [{
        key    = "ai-workload"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]

      remote_access = {
        ec2_ssh_key = aws_key_pair.eks_nodes.key_name
      }
    }

    memory = {
      name = "memory-optimized"

      instance_types = var.node_groups.memory.instance_types
      ami_type       = "AL2_x86_64"
      
      min_size     = var.node_groups.memory.min_size
      max_size     = var.node_groups.memory.max_size
      desired_size = var.node_groups.memory.desired_size

      disk_size = 100
      disk_type = "gp3"

      labels = {
        Environment = var.environment
        NodeGroup   = "memory-optimized"
        workload    = "high-memory"
      }

      taints = [{
        key    = "high-memory"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]

      remote_access = {
        ec2_ssh_key = aws_key_pair.eks_nodes.key_name
      }
    }
  }

  # Fargate profiles
  fargate_profiles = {
    default = {
      name = "default"
      selectors = [
        {
          namespace = "enterprise-ai-gateway"
          labels = {
            fargate = "enabled"
          }
        }
      ]

      tags = {
        Environment = var.environment
      }

      timeouts = {
        create = "20m"
        delete = "20m"
      }
    }
  }

  # aws-auth configmap
  manage_aws_auth_configmap = true

  aws_auth_roles = [
    {
      rolearn  = aws_iam_role.eks_admin.arn
      username = "eks-admin"
      groups   = ["system:masters"]
    }
  ]

  aws_auth_users = var.aws_auth_users
  aws_auth_accounts = var.aws_auth_accounts

  tags = {
    Environment = var.environment
  }
}

# KMS Key for EKS
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name = "${var.cluster_name}-eks-encryption-key"
  }
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${var.cluster_name}-eks-encryption-key"
  target_key_id = aws_kms_key.eks.key_id
}

# Key pair for EKS nodes
resource "tls_private_key" "eks_nodes" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "eks_nodes" {
  key_name   = "${var.cluster_name}-eks-nodes"
  public_key = tls_private_key.eks_nodes.public_key_openssh

  tags = {
    Name = "${var.cluster_name}-eks-nodes"
  }
}

# Store private key in AWS Secrets Manager
resource "aws_secretsmanager_secret" "eks_nodes_private_key" {
  name                    = "${var.cluster_name}-eks-nodes-private-key"
  description             = "Private key for EKS nodes"
  recovery_window_in_days = 7

  tags = {
    Name = "${var.cluster_name}-eks-nodes-private-key"
  }
}

resource "aws_secretsmanager_secret_version" "eks_nodes_private_key" {
  secret_id     = aws_secretsmanager_secret.eks_nodes_private_key.id
  secret_string = tls_private_key.eks_nodes.private_key_pem
}

# IAM role for EKS administration
resource "aws_iam_role" "eks_admin" {
  name = "${var.cluster_name}-eks-admin"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_admin_policy" {
  role       = aws_iam_role.eks_admin.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

# Security group for additional rules
resource "aws_security_group" "additional" {
  name_prefix = "${var.cluster_name}-additional"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = [
      "10.0.0.0/8",
      "172.16.0.0/12",
      "192.168.0.0/16",
    ]
  }

  tags = {
    Name = "${var.cluster_name}-additional"
  }
}