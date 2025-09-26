#!/bin/bash

# CUDNT Automated Deployment Script
# Consciousness Mathematics Framework Deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Consciousness Mathematics Constants
PHI="1.618034"
CONSCIOUSNESS_RATIO="3.761905"

echo -e "${BLUE}🚀 CUDNT Deployment Automation Script${NC}"
echo -e "${BLUE}=====================================${NC}"
echo -e "${YELLOW}Golden Ratio (φ): ${PHI}${NC}"
echo -e "${YELLOW}Consciousness Ratio: ${CONSCIOUSNESS_RATIO}${NC}"
echo ""

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
NAMESPACE="cudnt-${ENVIRONMENT}"
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"ghcr.io/your-org"}
VERSION=${VERSION:-"latest"}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check required tools
    for tool in kubectl helm docker terraform; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is not installed"
            exit 1
        fi
    done

    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

setup_infrastructure() {
    log_info "Setting up infrastructure with Terraform..."

    cd terraform

    terraform init
    terraform plan -var="environment=${ENVIRONMENT}" -out=tfplan
    terraform apply tfplan

    cd ..
    log_success "Infrastructure setup completed"
}

build_and_push_images() {
    log_info "Building and pushing Docker images..."

    # Build backend
    log_info "Building backend image..."
    docker build -t "${DOCKER_REGISTRY}/cudnt-backend:${VERSION}" ./backend
    docker push "${DOCKER_REGISTRY}/cudnt-backend:${VERSION}"

    # Build frontend
    log_info "Building frontend image..."
    docker build -t "${DOCKER_REGISTRY}/cudnt-frontend:${VERSION}" ./frontend
    docker push "${DOCKER_REGISTRY}/cudnt-frontend:${VERSION}"

    log_success "Docker images built and pushed"
}

setup_monitoring() {
    log_info "Setting up monitoring stack..."

    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

    # Install Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \\
        --namespace monitoring \\
        --values monitoring/prometheus-values.yaml \\
        --wait

    # Install Grafana dashboards
    kubectl apply -f monitoring/grafana-dashboard-configmap.yaml

    log_success "Monitoring stack deployed"
}

deploy_application() {
    log_info "Deploying CUDNT application..."

    # Create namespace
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

    # Apply secrets
    kubectl apply -f kubernetes/secrets/ -n ${NAMESPACE}

    # Deploy database
    log_info "Deploying database..."
    kubectl apply -f kubernetes/database/ -n ${NAMESPACE}

    # Wait for database to be ready
    kubectl wait --for=condition=ready pod -l app=mongodb -n ${NAMESPACE} --timeout=300s

    # Deploy backend
    log_info "Deploying backend..."
    envsubst < kubernetes/backend/deployment.yaml | kubectl apply -f - -n ${NAMESPACE}
    kubectl apply -f kubernetes/backend/service.yaml -n ${NAMESPACE}

    # Wait for backend to be ready
    kubectl wait --for=condition=available deployment/cudnt-backend -n ${NAMESPACE} --timeout=300s

    # Deploy frontend
    log_info "Deploying frontend..."
    envsubst < kubernetes/frontend/deployment.yaml | kubectl apply -f - -n ${NAMESPACE}
    kubectl apply -f kubernetes/frontend/service.yaml -n ${NAMESPACE}

    # Deploy ingress
    log_info "Deploying ingress..."
    kubectl apply -f kubernetes/ingress.yaml -n ${NAMESPACE}

    # Wait for frontend to be ready
    kubectl wait --for=condition=available deployment/cudnt-frontend -n ${NAMESPACE} --timeout=300s

    log_success "Application deployed successfully"
}

run_health_checks() {
    log_info "Running health checks..."

    # Wait for services to be ready
    sleep 30

    # Get service URLs
    BACKEND_URL=$(kubectl get ingress cudnt-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}')

    # Health check backend
    if curl -f "https://${BACKEND_URL}/api/health" &> /dev/null; then
        log_success "Backend health check passed"
    else
        log_error "Backend health check failed"
        exit 1
    fi

    # Check consciousness mathematics validation
    if curl -f "https://${BACKEND_URL}/api/consciousness/validate" &> /dev/null; then
        log_success "Consciousness mathematics validation passed"
    else
        log_warning "Consciousness mathematics validation failed"
    fi

    log_success "Health checks completed"
}

run_performance_tests() {
    log_info "Running performance tests..."

    # Install performance testing tools
    kubectl apply -f testing/performance-test-job.yaml -n ${NAMESPACE}

    # Wait for tests to complete
    kubectl wait --for=condition=complete job/cudnt-performance-test -n ${NAMESPACE} --timeout=600s

    # Collect results
    kubectl logs job/cudnt-performance-test -n ${NAMESPACE}

    log_success "Performance tests completed"
}

# Main deployment flow
main() {
    check_prerequisites
    setup_infrastructure
    build_and_push_images
    setup_monitoring
    deploy_application
    run_health_checks
    run_performance_tests

    echo ""
    log_success "🎉 CUDNT Enterprise Deployment Complete!"
    echo ""
    echo "🌐 Frontend: https://app.cudnt.com"
    echo "🔌 API: https://api.cudnt.com"
    echo "📊 Monitoring: https://monitoring.cudnt.com"
    echo "📈 Analytics: https://analytics.cudnt.com"
    echo ""
    echo "🚀 Consciousness Mathematics Framework Active"
    echo "   Golden Ratio (φ): ${PHI}"
    echo "   Consciousness Ratio: ${CONSCIOUSNESS_RATIO}"
    echo "   Performance Target: O(n²) → O(n^1.44)"
}

# Run main function
main "$@"
