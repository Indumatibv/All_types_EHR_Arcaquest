# ArcaQuest EHR Application - Deployment Files

## Directory Contents

This directory contains all the files needed to deploy the arcaquest-ehr application to Azure Kubernetes Service (AKS).

### Application Files

- **main.py** - Main FastAPI application code (fixed for LangChain v1.0)
- **requirements.txt** - Python dependencies (includes langchain-openai)
- **Dockerfile** - Docker image build configuration
- **ehr_transcript.txt** - Sample transcript file
- **summary.txt** - Sample summary file

### Kubernetes Configuration Files

- **deployment.yaml** - Kubernetes Deployment configuration
- **service.yaml** - Kubernetes Service configuration (LoadBalancer)
- **secret.yaml** - Template for Kubernetes Secret (Azure OpenAI key)

### Documentation

- **AKS_Complete_Deployment_Fix_Guide.md** - Complete troubleshooting and deployment guide

## Quick Start

### Prerequisites

1. Azure CLI installed and logged in
2. kubectl installed and configured
3. Access to Azure subscription and AKS cluster

### Build and Push Image

```bash
# Build image using Azure Container Registry
az acr build \
  --registry arcaquest \
  --image arcaquest-ehr:latest \
  --platform linux/amd64 \
  .
```

### Deploy to Kubernetes

```bash
# Create secret (replace with your actual key)
kubectl create secret generic azure-openai-secret \
  --from-literal=AZURE_OPENAI_KEY=your-key-here

# Apply Kubernetes configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check deployment status
kubectl get pods
kubectl get service ehr-app-service
```

## Image Information

**Registry:** arcaquest.azurecr.io
**Repository:** arcaquest-ehr
**Tag:** latest
**Platform:** linux/amd64

## Application Endpoints

- **Health Check:** `GET /health`
- **Process EHR:** `POST /process-ehr-json/`
- **API Docs:** `GET /docs` (FastAPI auto-generated)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| AZURE_OPENAI_KEY | Yes | Azure OpenAI API key |
| AZURE_OPENAI_API_VERSION | Yes | API version (default: 2024-12-01-preview) |
| MAX_CONCURRENT_REQUESTS | No | Max concurrent requests (default: 5) |

## Troubleshooting

See `AKS_Complete_Deployment_Fix_Guide.md` for detailed troubleshooting steps.

### Common Issues

1. **ImagePullBackOff** - Check image architecture and ACR permissions
2. **CrashLoopBackOff** - Check application logs with `kubectl logs <pod-name>`
3. **Service not accessible** - Check LoadBalancer external IP with `kubectl get svc`

## Support

For issues or questions, refer to the complete deployment guide.
