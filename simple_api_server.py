#!/usr/bin/env python3
"""
Simple chAIos API Server
========================
Lightweight API server for testing and development
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our custom systems
try:
    from cudnt_universal_accelerator import get_cudnt_accelerator
    from simple_redis_alternative import get_redis_client
    from simple_postgresql_alternative import get_postgres_client
    from performance_optimization_engine import PerformanceOptimizationEngine, PerformanceConfig
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Systems not available: {e}")
    SYSTEMS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="chAIos Simple API Server",
    description="Lightweight API server with CUDNT, Redis, and PostgreSQL alternatives",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
if SYSTEMS_AVAILABLE:
    cudnt = get_cudnt_accelerator()
    redis_client = get_redis_client()
    db_client = get_postgres_client()
    
    # Simple performance config without monitoring
    perf_config = PerformanceConfig(
        enable_gpu_acceleration=True,
        enable_redis_caching=True,
        enable_database_optimization=True,
        enable_compression=False,
        enable_monitoring=False
    )
    perf_engine = PerformanceOptimizationEngine(perf_config)

# Pydantic models
class ConsciousnessRequest(BaseModel):
    data: List[List[float]]
    algorithm: str = "matrix_optimization"
    enhancement_level: float = 1.618

class QuantumRequest(BaseModel):
    qubits: int = 8
    iterations: int = 100

class CacheRequest(BaseModel):
    key: str
    value: Optional[Any] = None
    ttl: int = 3600

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "chAIos Simple API Server",
        "status": "operational",
        "systems_available": SYSTEMS_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "cudnt": "/cudnt/info",
            "cache": "/cache/{key}",
            "database": "/database/stats",
            "prime aligned compute": "/prime aligned compute/process",
            "quantum": "/quantum/simulate"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "systems": {}
    }
    
    if SYSTEMS_AVAILABLE:
        try:
            # Test CUDNT
            cudnt_info = cudnt.get_acceleration_info()
            status["systems"]["cudnt"] = {
                "status": "operational",
                "name": cudnt_info["full_name"],
                "features": len([f for f, v in cudnt_info["features"].items() if v])
            }
            
            # Test Redis
            redis_client.set("health_check", "ok", ex=60)
            redis_result = redis_client.get("health_check")
            status["systems"]["redis"] = {
                "status": "operational" if redis_result == "ok" else "error",
                "test_result": redis_result
            }
            
            # Test Database
            db_stats = db_client.get_database_stats()
            status["systems"]["database"] = {
                "status": "operational",
                "records": db_stats
            }
            
        except Exception as e:
            status["systems"]["error"] = str(e)
            status["status"] = "degraded"
    
    return status

@app.get("/cudnt/info")
async def cudnt_info():
    """Get CUDNT acceleration information"""
    if not SYSTEMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Systems not available")
    
    try:
        info = cudnt.get_acceleration_info()
        return {
            "status": "success",
            "cudnt_info": info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CUDNT error: {str(e)}")

@app.post("/prime aligned compute/process")
async def process_consciousness(request: ConsciousnessRequest):
    """Process prime aligned compute data with CUDNT"""
    if not SYSTEMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Systems not available")
    
    try:
        import numpy as np
        
        # Convert data to numpy array
        data_array = np.array(request.data)
        
        # Process with CUDNT
        result = cudnt.accelerate_quantum_computing(data_array, 100)
        
        # Store in database
        db_client.insert_consciousness_data(
            request.data,
            request.algorithm,
            request.enhancement_level,
            result.get("processing_time", 0.0)
        )
        
        return {
            "status": "success",
            "result": result,
            "algorithm": request.algorithm,
            "enhancement_level": request.enhancement_level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prime aligned compute processing error: {str(e)}")

@app.post("/quantum/simulate")
async def simulate_quantum(request: QuantumRequest):
    """Simulate quantum computing with CUDNT"""
    if not SYSTEMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Systems not available")
    
    try:
        import numpy as np
        
        # Generate test data
        test_data = np.random.random(2**request.qubits)
        
        # Process with CUDNT
        result = cudnt.accelerate_quantum_computing(test_data, request.iterations)
        
        # Store quantum result
        db_client.insert_quantum_result(
            request.qubits,
            request.iterations,
            result.get("fidelity", 0.95),
            result.get("processing_time", 0.0),
            "CUDNT"
        )
        
        return {
            "status": "success",
            "quantum_simulation": {
                "qubits": request.qubits,
                "iterations": request.iterations,
                "result": result
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantum simulation error: {str(e)}")

@app.get("/cache/{key}")
async def get_cache(key: str):
    """Get value from cache"""
    if not SYSTEMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Systems not available")
    
    try:
        value = redis_client.get(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        
        return {
            "status": "success",
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache error: {str(e)}")

@app.post("/cache")
async def cache_operation(request: CacheRequest):
    """Set or get value from cache"""
    if not SYSTEMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Systems not available")

    try:
        if request.value is not None:
            # Set operation
            redis_client.set(request.key, request.value, ex=request.ttl)
            return {
                "status": "success",
                "operation": "set",
                "key": request.key,
                "ttl": request.ttl,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Get operation
            cached_value = redis_client.get(request.key)
            if cached_value is not None:
                return {
                    "status": "success",
                    "operation": "get",
                    "key": request.key,
                    "value": cached_value,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "not_found",
                    "operation": "get",
                    "key": request.key,
                    "message": "Key not found in cache",
                    "timestamp": datetime.now().isoformat()
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache error: {str(e)}")

@app.get("/database/stats")
async def database_stats():
    """Get database statistics"""
    if not SYSTEMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Systems not available")
    
    try:
        stats = db_client.get_database_stats()
        
        return {
            "status": "success",
            "database_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/performance/status")
async def performance_status():
    """Get performance optimization status"""
    if not SYSTEMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Systems not available")
    
    try:
        # Get basic performance info
        status = {
            "status": "success",
            "performance_engine": {
                "gpu_available": perf_engine.gpu_manager.gpu_available,
                "cache_connected": perf_engine.cache_manager.connected,
                "database_connected": perf_engine.db_optimizer.db_client is not None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance status error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting chAIos Simple API Server...")
    print("=" * 50)
    
    if SYSTEMS_AVAILABLE:
        print("✅ All systems available:")
        print(f"   🚀 CUDNT: {cudnt.get_acceleration_info()['full_name']}")
        print(f"   💾 Redis: Connected")
        print(f"   🗄️ Database: Connected")
        print(f"   ⚡ Performance Engine: Ready")
    else:
        print("⚠️ Some systems not available")
    
    print("\n🌐 Server starting on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
