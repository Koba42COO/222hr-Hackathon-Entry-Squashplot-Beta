#!/bin/bash

# 🚀 Consciousness Framework Deployment Script
# This script deploys all tools and services in the dev ecosystem

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Function to build all Docker images
build_images() {
    print_status "Building all Docker images..."
    docker-compose build --parallel
    print_success "All Docker images built successfully"
}

# Function to start all services
start_services() {
    print_status "Starting all services..."
    docker-compose up -d
    print_success "All services started successfully"
}

# Function to stop all services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped successfully"
}

# Function to show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
}

# Function to view logs
view_logs() {
    if [ -z "$2" ]; then
        print_status "Viewing logs for all services..."
        docker-compose logs -f
    else
        print_status "Viewing logs for service: $2"
        docker-compose logs -f "$2"
    fi
}

# Function to restart specific service
restart_service() {
    if [ -z "$2" ]; then
        print_error "Please specify a service name to restart"
        echo "Available services:"
        docker-compose config --services
        exit 1
    fi

    print_status "Restarting service: $2"
    docker-compose restart "$2"
    print_success "Service $2 restarted successfully"
}

# Function to update all services
update_services() {
    print_status "Updating all services..."
    docker-compose pull
    docker-compose build --parallel
    docker-compose up -d
    print_success "All services updated successfully"
}

# Function to clean up unused resources
cleanup() {
    print_status "Cleaning up unused Docker resources..."
    docker system prune -f
    docker volume prune -f
    print_success "Cleanup completed"
}

# Function to show available commands
show_help() {
    echo "🚀 Consciousness Framework Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build      Build all Docker images"
    echo "  start      Start all services"
    echo "  stop       Stop all services"
    echo "  restart    Restart all services"
    echo "  status     Show status of all services"
    echo "  logs       View logs for all services"
    echo "  logs SERVICE View logs for specific service"
    echo "  update     Update all services with latest code"
    echo "  cleanup    Clean up unused Docker resources"
    echo "  full       Full deployment (build + start)"
    echo "  help       Show this help message"
    echo ""
    echo "Available services:"
    docker-compose config --services 2>/dev/null || echo "  (Run 'build' first to see services)"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_docker
        build_images
        ;;
    "start")
        check_docker
        start_services
        ;;
    "stop")
        check_docker
        stop_services
        ;;
    "restart")
        check_docker
        if [ -z "$2" ]; then
            print_status "Restarting all services..."
            docker-compose restart
            print_success "All services restarted successfully"
        else
            restart_service "$@"
        fi
        ;;
    "status")
        check_docker
        show_status
        ;;
    "logs")
        check_docker
        view_logs "$@"
        ;;
    "update")
        check_docker
        update_services
        ;;
    "cleanup")
        check_docker
        cleanup
        ;;
    "full")
        check_docker
        print_status "Starting full deployment..."
        build_images
        start_services
        print_success "Full deployment completed!"
        echo ""
        print_status "Services are now running. Access them at:"
        echo "  SCADDA Energy:     http://localhost:8081"
        echo "  SCADDA Power:      http://localhost:8082"
        echo "  SquashPlot Chia:   http://localhost:8083"
        echo "  F2 GPU Optimizer:  http://localhost:8084"
        echo "  Consciousness:     http://localhost:8085"
        echo "  Unified Graph:     http://localhost:8086"
        echo "  Benchmark:         http://localhost:8087"
        echo "  Chia Analysis:     http://localhost:8088"
        echo "  Decentralized API: http://localhost:3002"
        echo "  Contribution API:  http://localhost:3003"
        echo "  CAT Credits API:   http://localhost:3004"
        echo "  Social Pub/Sub:    http://localhost:3005"
        ;;
    "help"|*)
        show_help
        ;;
esac
