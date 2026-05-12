# AWS Deployment Guide

This guide explains how to deploy the Whisperbox application to AWS for better scalability and performance.

## Architecture Overview

The AWS deployment uses the following components:

1. **Amazon ECR** - For storing Docker images
2. **Amazon ECS/Fargate** - For running containerized applications
3. **Amazon S3** - For storing audio/video files and transcripts
4. **Amazon API Gateway** - For exposing HTTP endpoints
5. **Amazon CloudWatch** - For logging and monitoring

![AWS Architecture Diagram](../images/aws-architecture.png)

## Prerequisites

- AWS CLI installed and configured
- Docker installed locally
- Basic understanding of AWS services
- An AWS account with permissions to create the necessary resources

## Deployment Steps

### 1. Create an ECR Repository

```bash
aws ecr create-repository --repository-name whisperbox
```

Note the repository URI from the output.

### 2. Build and Push the Docker Image

```bash
# Login to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com

# Build the image
docker build -t whisperbox .

# Tag the image
docker tag whisperbox:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/whisperbox:latest

# Push the image
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/whisperbox:latest
```

### 3. Create an S3 Bucket

```bash
aws s3 mb s3://whisperbox-files
```

### 4. Create ECS Cluster and Service

For simplicity, we recommend using the AWS Management Console to create an ECS cluster and service:

1. Go to the ECS console
2. Create a new cluster
3. Create a new task definition:
   - Use the Fargate launch type
   - Specify the ECR image URI
   - Configure CPU and memory (recommend at least 2vCPU and 4GB memory)
   - Add environment variables from your `.env` file
   - Add IAM roles for S3 access
4. Create a service in your cluster:
   - Use the task definition you created
   - Configure the number of tasks (instances) based on your needs
   - Set up a load balancer if needed

### 5. Configure API Gateway (Optional)

If you want to expose your ECS service through a managed API:

1. Create a new REST API in API Gateway
2. Create resources and methods to proxy requests to your ECS service
3. Deploy the API

## Environment Variables for AWS

When deploying to AWS, set the following environment variables in your ECS task definition:

```
WHISPER_MODEL=base
OUTPUT_FORMAT=txt
INCLUDE_DIARIZATION=false
FORCE_CPU=true
CACHE_ENABLED=true
HF_TOKEN=your_token_here
AWS_S3_BUCKET=whisperbox-files
```

## S3 Integration

To use S3 for file storage instead of local storage, modify the application to:

1. Upload files to S3 before processing
2. Download files from S3 when needed
3. Upload transcripts to S3 after processing

You can implement this by adding an S3 client to your application and modifying the file operations.

## Scaling Considerations

- **Vertical Scaling**: Increase the CPU and memory allocation in your task definition
- **Horizontal Scaling**: Increase the number of tasks in your ECS service
- **Spot Instances**: Use Fargate Spot for cost savings on non-critical workloads
- **Auto Scaling**: Configure ECS service auto scaling based on CPU/memory usage

## Cost Optimization

- Use Fargate Spot for non-critical workloads
- Scale down to zero when not in use
- Use smaller Whisper models for lower resource consumption
- Cache processed files in S3 to avoid reprocessing

## Monitoring and Logging

- CloudWatch Logs for container logs
- CloudWatch Metrics for performance monitoring
- CloudWatch Alarms for notifications on issues
- X-Ray for request tracing (optional)

## Security Considerations

- Use IAM roles with least privilege
- Encrypt S3 buckets with SSE-S3 or KMS
- Use security groups to restrict network access
- Store sensitive environment variables in AWS Secrets Manager