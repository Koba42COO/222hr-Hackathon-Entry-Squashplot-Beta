#!/usr/bin/env python3
"""
Enhanced chAIos API Server with Performance Optimizations
=========================================================
High-performance API server with GPU acceleration, caching, and monitoring
"""

import os
import time
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# Import performance optimization components
from performance_optimization_engine import (
    PerformanceOptimizationEngine, 
    PerformanceConfig,
    performance_engine
)

# Import existing prime aligned compute modules
try:
    from proper_consciousness_mathematics import ConsciousnessMathFramework
    from curated_tools_integration import get_curated_tools
    CONSCIOUSNESS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ prime aligned compute modules not available: {e}")
    CONSCIOUSNESS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance configuration
PERF_CONFIG = PerformanceConfig(
    enable_gpu_acceleration=True,
    enable_redis_caching=True,
    enable_database_optimization=True,
    enable_compression=True,
    enable_monitoring=False  # Disable monitoring to prevent CPU issues
)

# Initialize performance engine
perf_engine = PerformanceOptimizationEngine(PERF_CONFIG)

# FastAPI app with performance optimizations
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Enhanced chAIos API Server...")
    
    # Initialize performance optimizations
    await perf_engine.optimize_system()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced chAIos API Server...")

app = FastAPI(
    title="Enhanced chAIos API Server",
    description="High-performance prime aligned compute computing platform with GPU acceleration",
    version="2.0.0",
    lifespan=lifespan
)

# Add performance middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

# Pydantic models
class ProcessingRequest(BaseModel):
    data: Union[str, List[Any], Dict[str, Any]]
    algorithm: str = "prime_aligned_enhanced"
    parameters: Dict[str, Any] = {}
    use_gpu: bool = True
    use_cache: bool = True

class PerformanceRequest(BaseModel):
    test_type: str = "comprehensive"
    iterations: int = 1000
    use_gpu: bool = True

class CacheRequest(BaseModel):
    key: str
    value: Any
    ttl: int = 3600

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication dependency"""
    if credentials.credentials == "benchmark_token":
        return {"user_id": "benchmark_user", "permissions": ["read", "write", "system"]}
    elif credentials.credentials == "admin_token":
        return {"user_id": "admin", "permissions": ["read", "write", "system", "admin"]}
    else:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# Performance monitoring endpoint
@app.get("/performance/status")
async def get_performance_status():
    """Get current performance status"""
    try:
        summary = await perf_engine.get_performance_summary()
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "performance": summary
        }
    except Exception as e:
        logger.error(f"Performance status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# GPU acceleration endpoint
@app.post("/performance/gpu-test")
async def test_gpu_acceleration(request: PerformanceRequest):
    """Test GPU acceleration capabilities"""
    try:
        import numpy as np
        
        # Generate test data
        test_data = np.random.random((1000, 1000))
        
        # Run GPU-accelerated processing
        start_time = time.time()
        result = await perf_engine.gpu_manager.gpu_quantum_processing(
            test_data, 
            request.iterations
        )
        processing_time = time.time() - start_time
        
        return {
            "status": "completed",
            "processing_time": processing_time,
            "iterations": request.iterations,
            "gpu_available": perf_engine.gpu_manager.gpu_available,
            "gpu_info": {
                "type": result.get("acceleration", "CPU_OPTIMIZED"),
                "available": perf_engine.gpu_manager.gpu_available
            },
            "result": {
                "acceleration": result.get("acceleration", "CPU_OPTIMIZED"),
                "operations": result.get("operations", request.iterations),
                "processing_efficiency": result.get("processing_efficiency", 0.0)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"GPU test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoints
@app.post("/cache/set")
async def set_cache(request: CacheRequest):
    """Set cache value"""
    try:
        success = await perf_engine.cache_manager.set(
            request.key, 
            request.value, 
            request.ttl
        )
        
        return {
            "status": "success" if success else "failed",
            "key": request.key,
            "ttl": request.ttl,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache set error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/get/{key}")
async def get_cache(key: str):
    """Get cache value"""
    try:
        value = await perf_engine.cache_manager.get(key)
        
        return {
            "status": "found" if value is not None else "not_found",
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced prime aligned compute processing endpoint
@app.post("/prime aligned compute/process")
async def process_consciousness(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Enhanced prime aligned compute processing with GPU acceleration and caching"""
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = f"consciousness_{hash(str(request.data))}_{request.algorithm}"
        cached_result = None
        
        if request.use_cache:
            cached_result = await perf_engine.cache_manager.get(cache_key)
            if cached_result:
                return {
                    "status": "completed",
                    "result": cached_result,
                    "source": "cache",
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
        
        # Process with prime aligned compute mathematics
        if CONSCIOUSNESS_AVAILABLE:
            cmf = ConsciousnessMathFramework()
            
            # Apply prime aligned compute enhancement
            if isinstance(request.data, (list, dict)):
                processed_data = cmf.enhance_consciousness(request.data)
            else:
                processed_data = cmf.enhance_consciousness([request.data])
            
            # Apply GPU acceleration if available
            if request.use_gpu and perf_engine.gpu_manager.gpu_available:
                import numpy as np
                if isinstance(processed_data, list):
                    data_array = np.array(processed_data)
                    gpu_result = await perf_engine.gpu_manager.gpu_quantum_processing(
                        data_array, 
                        request.parameters.get('iterations', 100)
                    )
                    processed_data = gpu_result.get('result', processed_data)
            
            result = {
                "processed_data": processed_data,
                "consciousness_enhancement": 1.618,
                "algorithm": request.algorithm,
                "gpu_accelerated": request.use_gpu and perf_engine.gpu_manager.gpu_available,
                "processing_efficiency": len(processed_data) / (time.time() - start_time)
            }
            
        else:
            # Fallback processing
            result = {
                "processed_data": request.data,
                "consciousness_enhancement": 1.0,
                "algorithm": "fallback",
                "gpu_accelerated": False,
                "processing_efficiency": 1.0
            }
        
        # Cache result
        if request.use_cache:
            background_tasks.add_task(
                perf_engine.cache_manager.set,
                cache_key,
                result,
                3600  # 1 hour TTL
            )
        
        processing_time = time.time() - start_time
        
        return {
            "status": "completed",
            "result": result,
            "source": "processing",
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"prime aligned compute processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced tool execution endpoint
@app.post("/tools/execute")
async def execute_tool(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Execute prime aligned compute tools with performance optimizations"""
    start_time = time.time()
    
    try:
        # Get available tools
        if CONSCIOUSNESS_AVAILABLE:
            tools = get_curated_tools()
            tool_name = request.parameters.get('tool_name', 'grok_consciousness_coding')
            
            if tool_name in tools:
                tool_func = tools[tool_name]
                
                # Prepare parameters
                params = request.parameters.copy()
                params.pop('tool_name', None)
                
                # Execute tool with performance monitoring
                result = tool_func(**params)
                
                # Apply GPU acceleration if available
                if request.use_gpu and perf_engine.gpu_manager.gpu_available:
                    # Enhance result with GPU processing
                    if isinstance(result, dict) and 'result' in result:
                        import numpy as np
                        if isinstance(result['result'], (list, tuple)):
                            data_array = np.array(result['result'])
                            gpu_result = await perf_engine.gpu_manager.gpu_quantum_processing(
                                data_array, 
                                100
                            )
                            result['gpu_enhanced'] = gpu_result
                
                processing_time = time.time() - start_time
                
                return {
                    "status": "completed",
                    "tool": tool_name,
                    "result": result,
                    "processing_time": processing_time,
                    "gpu_accelerated": request.use_gpu and perf_engine.gpu_manager.gpu_available,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        else:
            raise HTTPException(status_code=503, detail="prime aligned compute tools not available")
            
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System optimization endpoint
@app.post("/system/optimize")
async def optimize_system(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Run system optimization"""
    try:
        # Run optimization in background
        background_tasks.add_task(perf_engine.optimize_system)
        
        return {
            "status": "optimization_started",
            "message": "System optimization running in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check with performance metrics"""
    try:
        # Get performance metrics
        summary = await perf_engine.get_performance_summary()
        
        # Determine health status
        cpu_usage = summary['system_health']['cpu_usage']
        memory_usage = summary['system_health']['memory_usage']
        
        if cpu_usage > 90 or memory_usage > 90:
            status = "degraded"
        elif cpu_usage > 70 or memory_usage > 70:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "performance": summary,
            "services": {
                "api": "operational",
                "gpu": "available" if perf_engine.gpu_manager.gpu_available else "unavailable",
                "cache": "connected" if perf_engine.cache_manager.connected else "disconnected",
                "database": "optimized" if perf_engine.db_optimizer.connection_pool else "basic"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Enhanced chAIos API Server",
        "version": "2.0.0",
        "status": "operational",
        "features": {
            "gpu_acceleration": perf_engine.gpu_manager.gpu_available,
            "redis_caching": perf_engine.cache_manager.connected,
            "database_optimization": perf_engine.db_optimizer.connection_pool is not None,
            "performance_monitoring": True,
            "consciousness_processing": CONSCIOUSNESS_AVAILABLE
        },
        "endpoints": {
            "performance": "/performance/status",
            "gpu_test": "/performance/gpu-test",
            "prime aligned compute": "/prime aligned compute/process",
            "tools": "/tools/execute",
            "cache": "/cache/{get,set}",
            "health": "/health"
        },
        "timestamp": datetime.now().isoformat()
    }

# Plugin catalog endpoint (for compatibility)
@app.get("/plugin/catalog")
async def get_plugin_catalog():
    """Get available plugins/tools"""
    try:
        if CONSCIOUSNESS_AVAILABLE:
            tools = get_curated_tools()
            return {
                "tools": list(tools.keys()),
                "count": len(tools),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "tools": [],
                "count": 0,
                "error": "prime aligned compute tools not available",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Plugin catalog error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the enhanced server
    uvicorn.run(
        "enhanced_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
