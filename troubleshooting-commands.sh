#!/bin/bash

# ArcaQuest EHR - Troubleshooting Commands
# Quick reference for common debugging commands

echo "=========================================="
echo "ArcaQuest EHR - Troubleshooting Commands"
echo "=========================================="

# Colors
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\n${YELLOW}=== POD STATUS ===${NC}"
echo "kubectl get pods"
echo "kubectl get pods -w  # watch mode"
kubectl get pods -l app=ehr-app

echo -e "\n${YELLOW}=== POD DETAILS ===${NC}"
POD_NAME=$(kubectl get pods -l app=ehr-app -o jsonpath='{.items[0].metadata.name}')
if [ -n "$POD_NAME" ]; then
    echo "Pod name: $POD_NAME"
    echo ""
    echo "To describe pod:"
    echo "kubectl describe pod $POD_NAME"
    echo ""
    echo "To view logs:"
    echo "kubectl logs $POD_NAME"
    echo "kubectl logs $POD_NAME --follow  # stream logs"
    echo "kubectl logs $POD_NAME --previous  # previous container"
fi

echo -e "\n${YELLOW}=== DEPLOYMENT STATUS ===${NC}"
echo "kubectl get deployment ehr-app-deployment"
kubectl get deployment ehr-app-deployment

echo -e "\n${YELLOW}=== SERVICE STATUS ===${NC}"
echo "kubectl get service ehr-app-service"
kubectl get service ehr-app-service

echo -e "\n${YELLOW}=== EVENTS ===${NC}"
echo "kubectl get events --sort-by='.lastTimestamp' | grep ehr-app"
kubectl get events --sort-by='.lastTimestamp' | grep ehr-app | tail -10

echo -e "\n${YELLOW}=== IMAGE IN ACR ===${NC}"
echo "az acr repository show-tags --name arcaquest --repository arcaquest-ehr --output table"

echo -e "\n${YELLOW}=== USEFUL COMMANDS ===${NC}"
echo ""
echo "# Port forward to test locally:"
echo "kubectl port-forward $POD_NAME 8000:8000"
echo "# Then: curl http://localhost:8000/health"
echo ""
echo "# Exec into pod:"
echo "kubectl exec -it $POD_NAME -- /bin/bash"
echo ""
echo "# Restart deployment:"
echo "kubectl rollout restart deployment ehr-app-deployment"
echo ""
echo "# Delete pod (will recreate):"
echo "kubectl delete pod $POD_NAME"
echo ""
echo "# Check secret:"
echo "kubectl get secret azure-openai-secret"
echo ""
echo "# View complete pod YAML:"
echo "kubectl get pod $POD_NAME -o yaml"
