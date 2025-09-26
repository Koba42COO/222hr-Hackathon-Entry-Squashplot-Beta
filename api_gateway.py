#!/usr/bin/env python3
"""
Unified API Gateway
===================
Central routing and orchestration system for all chAIos platform services
Provides unified API endpoints, load balancing, authentication, and service discovery.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac

from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trust_proxy import TrustProxyMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import httpx
import redis.asyncio as redis

from configuration_manager import ConfigurationManager, PlatformConfig

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    url: str
    health_check_url: Optional[str] = None
    timeout: float = 30.0
    retries: int = 3
    circuit_breaker_threshold: int = 5
    rate_limit: Optional[int] = None
    authentication_required: bool = False
    allowed_methods: List[str] = None

    def __post_init__(self):
        if self.allowed_methods is None:
            self.allowed_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]

@dataclass
class GatewayMetrics:
    """Gateway performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    service_requests: Dict[str, int] = None
    error_rate: float = 0.0

    def __post_init__(self):
        if self.service_requests is None:
            self.service_requests = {}

class APIGateway:
    """Unified API Gateway for the chAIos platform"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = config_manager.current_config

        # FastAPI app
        self.app = FastAPI(
            title="chAIos Polymath Brain Platform API",
            description="Unified API gateway for all platform services",
            version=self.config.version,
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Service registry
        self.services: Dict[str, ServiceEndpoint] = {}
        self.service_health: Dict[str, ServiceStatus] = {}
        self.circuit_breakers: Dict[str, int] = {}

        # HTTP client for service communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Redis for caching and rate limiting
        self.redis_client = None

        # Metrics
        self.metrics = GatewayMetrics()

        # Security
        self.security = HTTPBearer(auto_error=False)

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._register_services()

        # Start background tasks
        self._start_background_tasks()

    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.security.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Trust proxy for rate limiting
        self.app.add_middleware(TrustProxyMiddleware)

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health")
        async def health_check():
            """Gateway health check"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": self.config.version,
                "services": len(self.services)
            }

        @self.app.get("/status")
        async def system_status():
            """Complete system status"""
            return await self._get_system_status()

        @self.app.get("/metrics")
        async def gateway_metrics():
            """Gateway performance metrics"""
            return {
                "gateway_metrics": {
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                    "error_rate": self.metrics.error_rate,
                    "average_response_time": self.metrics.average_response_time,
                    "service_breakdown": self.metrics.service_requests
                },
                "service_health": {name: status.value for name, status in self.service_health.items()}
            }

        # Knowledge system routes
        @self.app.api_route("/knowledge/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def knowledge_proxy(request: Request, path: str, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._proxy_request("knowledge_rag", request, path, credentials)

        # AI/ML system routes
        @self.app.api_route("/ai/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def ai_proxy(request: Request, path: str, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._proxy_request("cudnt_accelerator", request, path, credentials)

        @self.app.api_route("/quantum/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def quantum_proxy(request: Request, path: str, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._proxy_request("quantum_simulator", request, path, credentials)

        # Polymath brain routes
        @self.app.api_route("/polymath/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def polymath_proxy(request: Request, path: str, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._proxy_request("polymath_brain", request, path, credentials)

        # Learning system routes
        @self.app.api_route("/learning/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def learning_proxy(request: Request, path: str, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._proxy_request("learning_pathways", request, path, credentials)

        # Authentication routes
        @self.app.post("/auth/login")
        async def auth_proxy(request: Request):
            return await self._proxy_request("auth_service", request, "login")

        @self.app.post("/auth/verify")
        async def auth_verify_proxy(request: Request, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._proxy_request("auth_service", request, "verify", credentials)

        # Unified search endpoint
        @self.app.get("/search")
        async def unified_search(request: Request, q: str, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Unified search across all knowledge systems"""
            return await self._unified_search(q, credentials)

        # Polymath query endpoint
        @self.app.post("/query")
        async def polymath_query(request: Request, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Advanced polymath query processing"""
            return await self._polymath_query(request, credentials)

    def _register_services(self):
        """Register all platform services"""

        # Main API services
        if self.config_manager.is_service_enabled('main_api'):
            self.services['main_api'] = ServiceEndpoint(
                name='Main API',
                url=f"http://localhost:{self.config.services['main_api'].port}",
                health_check_url=f"http://localhost:{self.config.services['main_api'].port}/health"
            )

        if self.config_manager.is_service_enabled('enhanced_api'):
            self.services['enhanced_api'] = ServiceEndpoint(
                name='Enhanced API',
                url=f"http://localhost:{self.config.services['enhanced_api'].port}",
                health_check_url=f"http://localhost:{self.config.services['enhanced_api'].port}/health",
                authentication_required=True
            )

        if self.config_manager.is_service_enabled('auth_service'):
            self.services['auth_service'] = ServiceEndpoint(
                name='Authentication Service',
                url=f"http://localhost:{self.config.services['auth_service'].port}",
                health_check_url=f"http://localhost:{self.config.services['auth_service'].port}/health"
            )

        # Knowledge systems
        if self.config_manager.is_service_enabled('knowledge_rag'):
            self.services['knowledge_rag'] = ServiceEndpoint(
                name='Knowledge RAG System',
                url="http://localhost:8003",  # Assuming default port
                health_check_url="http://localhost:8003/health",
                authentication_required=True,
                rate_limit=100
            )

        if self.config_manager.is_service_enabled('polymath_brain'):
            self.services['polymath_brain'] = ServiceEndpoint(
                name='Polymath Brain Trainer',
                url="http://localhost:8004",
                health_check_url="http://localhost:8004/health",
                authentication_required=True,
                rate_limit=50
            )

        # AI/ML systems
        if self.config_manager.is_service_enabled('cudnt_accelerator'):
            self.services['cudnt_accelerator'] = ServiceEndpoint(
                name='CUDNT Universal Accelerator',
                url="http://localhost:8005",
                health_check_url="http://localhost:8005/health",
                authentication_required=True,
                rate_limit=20
            )

        if self.config_manager.is_service_enabled('quantum_simulator'):
            self.services['quantum_simulator'] = ServiceEndpoint(
                name='Quantum Simulator',
                url="http://localhost:8006",
                health_check_url="http://localhost:8006/health",
                authentication_required=True,
                rate_limit=10
            )

        # Learning systems
        if self.config_manager.is_service_enabled('learning_pathways'):
            self.services['learning_pathways'] = ServiceEndpoint(
                name='Learning Pathway System',
                url="http://localhost:8007",
                health_check_url="http://localhost:8007/health",
                authentication_required=True,
                rate_limit=30
            )

    def _start_background_tasks(self):
        """Start background tasks for health monitoring"""
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._metrics_updater())

    async def _health_monitor(self):
        """Monitor service health"""
        while True:
            for service_name, service in self.services.items():
                if service.health_check_url:
                    try:
                        response = await self.http_client.get(service.health_check_url, timeout=5.0)
                        if response.status_code == 200:
                            self.service_health[service_name] = ServiceStatus.HEALTHY
                            self.circuit_breakers[service_name] = 0  # Reset circuit breaker
                        else:
                            self.service_health[service_name] = ServiceStatus.UNHEALTHY
                            self.circuit_breakers[service_name] = self.circuit_breakers.get(service_name, 0) + 1
                    except Exception:
                        self.service_health[service_name] = ServiceStatus.UNHEALTHY
                        self.circuit_breakers[service_name] = self.circuit_breakers.get(service_name, 0) + 1
                else:
                    self.service_health[service_name] = ServiceStatus.UNKNOWN

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _metrics_updater(self):
        """Update gateway metrics"""
        while True:
            if self.metrics.total_requests > 0:
                self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
            await asyncio.sleep(60)  # Update every minute

    async def _proxy_request(self, service_name: str, request: Request, path: str = "",
                           credentials: Optional[HTTPAuthorizationCredentials] = None) -> Response:
        """Proxy request to backend service"""

        start_time = time.time()
        self.metrics.total_requests += 1

        # Check if service exists
        if service_name not in self.services:
            self.metrics.failed_requests += 1
            raise HTTPException(status_code=503, detail=f"Service {service_name} not available")

        service = self.services[service_name]

        # Check circuit breaker
        if self.circuit_breakers.get(service_name, 0) >= service.circuit_breaker_threshold:
            self.metrics.failed_requests += 1
            raise HTTPException(status_code=503, detail=f"Service {service_name} is currently unavailable")

        # Check authentication
        if service.authentication_required and not credentials:
            self.metrics.failed_requests += 1
            raise HTTPException(status_code=401, detail="Authentication required")

        # Rate limiting (simplified)
        if service.rate_limit and await self._check_rate_limit(request, service_name, service.rate_limit):
            self.metrics.failed_requests += 1
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Build target URL
        target_url = f"{service.url}/{path}".rstrip('/')
        if request.url.query:
            target_url += f"?{request.url.query}"

        try:
            # Forward request
            headers = dict(request.headers)
            headers.pop('host', None)  # Remove host header

            # Add authorization if provided
            if credentials:
                headers['Authorization'] = f"Bearer {credentials.credentials}"

            # Get request body
            body = await request.body()

            response = await self.http_client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                timeout=service.timeout
            )

            # Update metrics
            self.metrics.successful_requests += 1
            self.metrics.service_requests[service_name] = self.metrics.service_requests.get(service_name, 0) + 1

            # Calculate response time
            response_time = time.time() - start_time
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_requests - 1)) + response_time
            ) / self.metrics.total_requests

            # Return response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )

        except Exception as e:
            self.metrics.failed_requests += 1
            self.circuit_breakers[service_name] = self.circuit_breakers.get(service_name, 0) + 1
            logger.error(f"Request to {service_name} failed: {e}")
            raise HTTPException(status_code=503, detail=f"Service {service_name} error: {str(e)}")

    async def _check_rate_limit(self, request: Request, service_name: str, limit: int) -> bool:
        """Check rate limiting (simplified implementation)"""
        # This is a simplified rate limiting check
        # In production, you'd use Redis or similar
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:{service_name}:{client_ip}"

        # For now, just return False (no rate limiting)
        # In production, implement proper rate limiting logic
        return False

    async def _unified_search(self, query: str, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Perform unified search across all knowledge systems"""

        search_results = {
            'query': query,
            'timestamp': time.time(),
            'results': {},
            'total_results': 0
        }

        # Search across available services
        search_services = ['knowledge_rag', 'polymath_brain']

        for service_name in search_services:
            if service_name in self.services and self.service_health.get(service_name) == ServiceStatus.HEALTHY:
                try:
                    search_url = f"{self.services[service_name].url}/search"
                    headers = {'Authorization': f'Bearer {credentials.credentials}'} if credentials else {}

                    response = await self.http_client.get(
                        search_url,
                        params={'q': query},
                        headers=headers,
                        timeout=10.0
                    )

                    if response.status_code == 200:
                        results = response.json()
                        search_results['results'][service_name] = results
                        search_results['total_results'] += len(results.get('results', []))

                except Exception as e:
                    logger.warning(f"Search failed for {service_name}: {e}")
                    search_results['results'][service_name] = {'error': str(e)}

        return search_results

    async def _polymath_query(self, request: Request, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Process advanced polymath queries"""

        try:
            request_data = await request.json()
            query = request_data.get('query', '')

            if not query:
                raise HTTPException(status_code=400, detail="Query is required")

            # Route to polymath brain system
            if 'polymath_brain' in self.services and self.service_health.get('polymath_brain') == ServiceStatus.HEALTHY:
                polymath_url = f"{self.services['polymath_brain'].url}/query"
                headers = {'Authorization': f'Bearer {credentials.credentials}'} if credentials else {}

                response = await self.http_client.post(
                    polymath_url,
                    json=request_data,
                    headers=headers,
                    timeout=60.0  # Allow longer timeout for complex queries
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    raise HTTPException(status_code=response.status_code, detail="Polymath query failed")
            else:
                raise HTTPException(status_code=503, detail="Polymath brain service unavailable")

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Polymath query error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        status = {
            'gateway': {
                'status': 'healthy',
                'version': self.config.version,
                'uptime': time.time()  # Would track actual uptime in production
            },
            'services': {},
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'error_rate': self.metrics.error_rate,
                'average_response_time': self.metrics.average_response_time
            }
        }

        # Get status from each service
        for service_name, service in self.services.items():
            service_status = {
                'name': service.name,
                'url': service.url,
                'health': self.service_health.get(service_name, ServiceStatus.UNKNOWN).value,
                'circuit_breaker': self.circuit_breakers.get(service_name, 0),
                'requests': self.metrics.service_requests.get(service_name, 0)
            }

            # Try to get detailed status from service
            if service.health_check_url and self.service_health.get(service_name) == ServiceStatus.HEALTHY:
                try:
                    response = await self.http_client.get(service.health_check_url, timeout=5.0)
                    if response.status_code == 200:
                        service_status['details'] = response.json()
                except:
                    pass

            status['services'][service_name] = service_status

        return status

    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the API gateway"""

        logger.info(f"Starting API Gateway on {host}:{port}")
        logger.info(f"Environment: {self.config.environment.value}")
        logger.info(f"Services registered: {len(self.services)}")

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level=self.config.monitoring.log_level.lower()
        )

def main():
    """Main entry point for API Gateway"""

    import argparse

    parser = argparse.ArgumentParser(description='chAIos API Gateway')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--config-dir', default='config', help='Configuration directory')

    args = parser.parse_args()

    # Initialize configuration
    config_manager = ConfigurationManager(args.config_dir)

    # Validate configuration
    errors = config_manager.validate_configuration()
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  ❌ {error}")
        exit(1)

    # Create and run gateway
    gateway = APIGateway(config_manager)

    try:
        gateway.run(host=args.host, port=args.port, reload=args.reload)
    except KeyboardInterrupt:
        print("\n🛑 API Gateway stopped")
    except Exception as e:
        print(f"❌ API Gateway error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
