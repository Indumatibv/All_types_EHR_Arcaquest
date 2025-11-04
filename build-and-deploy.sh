#!/bin/bash

# ArcaQuest EHR - Build and Deploy Script
# This script builds the Docker image and deploys to AKS

set -e  # Exit on error

echo "=========================================="
echo "ArcaQuest EHR - Build and Deploy"
echo "=========================================="

# Configuration
REGISTRY="arcaquest"
IMAGE_NAME="arcaquest-ehr"
TAG="latest"
RESOURCE_GROUP="arcaquest"
AKS_CLUSTER="arcaquest-aks"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Login to Azure
echo -e "\n${YELLOW}Step 1: Logging in to Azure...${NC}"
az account show > /dev/null 2>&1 || az login

# Step 2: Set subscription
echo -e "\n${YELLOW}Step 2: Setting subscription...${NC}"
az account set --subscription "Microsoft Azure Sponsorship"
echo -e "${GREEN}✓ Subscription set${NC}"

# Step 3: Get AKS credentials
echo -e "\n${YELLOW}Step 3: Getting AKS credentials...${NC}"
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER --overwrite-existing
echo -e "${GREEN}✓ AKS credentials configured${NC}"

# Step 4: Build image
echo -e "\n${YELLOW}Step 4: Building Docker image in ACR...${NC}"
az acr build \
  --registry $REGISTRY \
  --image ${IMAGE_NAME}:${TAG} \
  --platform linux/amd64 \
  .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image built successfully${NC}"
else
    echo -e "${RED}✗ Image build failed${NC}"
    exit 1
fi

# Step 5: Check if secret exists
echo -e "\n${YELLOW}Step 5: Checking Kubernetes secret...${NC}"
if kubectl get secret azure-openai-secret > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Secret exists${NC}"
else
    echo -e "${YELLOW}⚠ Secret 'azure-openai-secret' not found${NC}"
    echo "Please create it with:"
    echo "kubectl create secret generic azure-openai-secret --from-literal=AZURE_OPENAI_KEY=your-key-here"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 6: Apply deployment
echo -e "\n${YELLOW}Step 6: Applying Kubernetes deployment...${NC}"
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
echo -e "${GREEN}✓ Deployment applied${NC}"

# Step 7: Restart deployment to pull new image
echo -e "\n${YELLOW}Step 7: Restarting deployment...${NC}"
kubectl rollout restart deployment ehr-app-deployment
echo -e "${GREEN}✓ Deployment restarted${NC}"

# Step 8: Wait for rollout
echo -e "\n${YELLOW}Step 8: Waiting for rollout to complete...${NC}"
kubectl rollout status deployment ehr-app-deployment --timeout=5m

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Rollout completed successfully${NC}"
else
    echo -e "${RED}✗ Rollout failed or timed out${NC}"
    echo "Check pod status with: kubectl get pods"
    echo "Check logs with: kubectl logs <pod-name>"
    exit 1
fi

# Step 9: Get pod status
echo -e "\n${YELLOW}Step 9: Checking pod status...${NC}"
kubectl get pods -l app=ehr-app

# Step 10: Get service info
echo -e "\n${YELLOW}Step 10: Getting service information...${NC}"
kubectl get service ehr-app-service

echo -e "\n${GREEN}=========================================="
echo "Deployment Complete!"
echo "==========================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Check pod logs: kubectl logs <pod-name>"
echo "2. Test health endpoint: curl http://<external-ip>/health"
echo "3. View API docs: http://<external-ip>/docs"

# Get external IP
EXTERNAL_IP=$(kubectl get service ehr-app-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$EXTERNAL_IP" ]; then
    echo -e "\n${GREEN}External IP: $EXTERNAL_IP${NC}"
    echo "Health check: http://$EXTERNAL_IP/health"
    echo "API docs: http://$EXTERNAL_IP/docs"
else
    echo -e "\n${YELLOW}External IP pending... Check with: kubectl get svc ehr-app-service${NC}"
fi
