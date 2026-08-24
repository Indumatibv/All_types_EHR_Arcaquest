# ArcaQuest Stream Processing - Architecture & Deployment Guide

This document outlines the architecture, infrastructure requirements, and deployment strategy for the ArcaQuest Stream Processing pipeline. It is designed to guide DevOps and backend engineers in deploying the solution to production on AWS Elastic Kubernetes Service (EKS).

---

## 1. Architectural Overview

The ArcaQuest Stream Processing pipeline is an asynchronous, event-driven microservice architecture that decouples HTTP ingestion from heavy LLM inference. 

The architecture consists of four primary components:

1. **Ingestion API (`api.py`)**
   * A lightweight Flask web server that acts as the "front door".
   * Instantly accepts streaming conversation chunks from the frontend UI and enqueues them into the message broker.
   * Responds immediately with HTTP `202 Accepted` to prevent frontend blocking or timeouts.

2. **Message Broker (ElastiCache Redis)**
   * Utilizes the existing Amazon ElastiCache Redis cluster (`ahs-ehr-dev-redis-rg`).
   * Manages the FIFO queue (`arcaquest_chunk_queue`) for conversation chunks.
   * Acts as a distributed state store for persistent patient session memory.

3. **Background Worker (`redis_worker.py` / `stream_processor.py`)**
   * A continuous Python daemon running alongside the API.
   * Polls the Redis queue, extracts chunks, and executes the semantic chunking/reranking logic using an external Qwen3 Reranker API.
   * Natively invokes the backend AWS Lambda function via the `boto3` SDK over a raw TCP connection, securely bypassing strict API Gateway timeouts.

4. **LLM Engine (AWS Lambda & Amazon Bedrock)**
   * An isolated AWS Lambda function (`bedrock_lambda.py`) deployed specifically to interact with the Amazon Bedrock Agent Runtime.
   * Offloads the heavy LLM inference (Amazon Nova Pro) into the serverless layer.

---

## 2. AWS Infrastructure Requirements

To deploy this architecture to the `AHS-EHR-Dev-arcaquest-eks` EKS cluster, the following AWS resources and permissions are required:

### A. AWS Lambda
* **Function:** Deploy `bedrock_lambda.py` as a Python 3.10+ Lambda function named `ArcaQuest_llm_service`.
* **Execution Role:** The Lambda's IAM Execution Role requires an inline policy granting `bedrock:InvokeAgent` and `bedrock-agentcore:InvokeAgentRuntime` permissions for the specific Bedrock Medical Assistant Agent ARN.
* **Timeout:** Set the Lambda execution timeout to at least **1 minute 0 seconds** to accommodate long-running Bedrock responses.

### B. EKS Cluster (IRSA)
* **Service Account:** The worker pods will run under the `ehr-app-sa` Kubernetes Service Account.
* **IAM Permission:** The IAM Role associated with `ehr-app-sa` **must** have the `lambda:InvokeFunction` permission attached for the `ArcaQuest_llm_service` Lambda function. This enables seamless, keyless `boto3` authentication.

### C. Amazon ElastiCache (Redis)
* **Endpoint:** Use the Primary write/read endpoint for the `ahs-ehr-dev-redis-rg` cluster.
* **Network:** The EKS worker nodes must have security group access to port `6379` on the ElastiCache VPC subnets.

---

## 3. Environment Variables

The deployment YAML for the EKS pods must inject the following environment variables:

| Variable | Description | Example / Default |
| :--- | :--- | :--- |
| `REDIS_URL` | The connection string for the ElastiCache cluster. Enables production Redis integration. | `redis://master.ahs-ehr-dev-redis-rg.whubdm.aps1.cache.amazonaws.com:6379/0` |
| `AWS_REGION` | The AWS region where the Lambda and SSM parameters reside. | `ap-south-1` |
| `SSM_NOVA_MODEL_PARAM` | (Optional) SSM Parameter for dynamic LLM configuration. | `/arcaquest/prod/nova_model_id` |

*Note: No `AWS_ACCESS_KEY_ID` or secret keys should be passed as environment variables in production, as authentication is handled by the EKS Service Account.*

---

## 4. Codebase Structure

* `api.py` - The Flask ingestion server.
* `redis_worker.py` - The daemon that polls Redis and orchestrates processing.
* `stream_processor.py` - Contains the core logic for the sliding window chunker, Reranker integration, and `boto3` Lambda invocation.
* `aws_mock.py` - A dual-purpose abstraction layer. Connects to real Redis via `redis-py` when `REDIS_URL` is present, or falls back to local file mocks (`.local_state/`) for local development.
* `config.py` - Dynamic configuration manager that fetches active models from AWS SSM Parameter Store.
* `field_handlers.py` - Utilities for parsing, normalizing, and flattening questionnaire payload schemas.
* `bedrock_lambda.py` - The standalone codebase for the AWS Lambda function.

---

## 5. Deployment Strategy (Next Steps)

To transition this codebase into the active EKS environment, the following DevOps steps must be completed:

1. **Dockerization:** 
   * Write a `Dockerfile` that packages `api.py`, `redis_worker.py`, and all dependencies from `requirements.txt`.
   * Consider using a process manager like `supervisord` to run both the API and Worker in the same container, or deploy them as separate microservices (recommended).
2. **Kubernetes Manifests:** 
   * Create `deployment.yaml` and `service.yaml` definitions.
   * Bind the pods to the `ehr-app-sa` Service Account.
3. **Lambda Deployment:**
   * Zip `bedrock_lambda.py` and deploy it to AWS Lambda. Configure the IAM role and timeout limit.
4. **CI/CD Integration:**
   * Inject the deployment into the standard pipeline.
