variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Base name for AWS resources"
  type        = string
  default     = "solarcast"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "task_cpu" {
  description = "CPU units for the ECS task definition"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Memory (MB) for the ECS task definition"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Number of ECS service tasks"
  type        = number
  default     = 1
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}
