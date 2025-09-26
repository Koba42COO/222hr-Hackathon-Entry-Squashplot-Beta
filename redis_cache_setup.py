#!/usr/bin/env python3
"""
Redis Cache Setup for chAIos Platform
=====================================
Sets up Redis caching system for high-performance data access
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisCacheSetup:
    """Redis cache setup and management"""
    
    def __init__(self):
        self.redis_client = None
        self.connected = False
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            import redis
            
            # Try to connect to Redis
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            logger.info("✅ Redis cache connected successfully")
            
        except ImportError:
            logger.warning("⚠️ Redis package not installed. Install with: pip install redis")
            self.connected = False
            self.redis_client = None
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
            self.connected = False
            self.redis_client = None
    
    async def setup_cache_structure(self) -> Dict[str, Any]:
        """Set up cache structure and initial data"""
        if not self.connected:
            return {"status": "redis_not_available"}
        
        try:
            # Set up cache keys and initial data
            cache_structure = {
                "consciousness_data": {
                    "ttl": 3600,  # 1 hour
                    "description": "prime aligned compute processing results"
                },
                "quantum_results": {
                    "ttl": 1800,  # 30 minutes
                    "description": "Quantum computing results"
                },
                "user_sessions": {
                    "ttl": 7200,  # 2 hours
                    "description": "User session data"
                },
                "api_responses": {
                    "ttl": 900,   # 15 minutes
                    "description": "Cached API responses"
                },
                "performance_metrics": {
                    "ttl": 300,   # 5 minutes
                    "description": "System performance metrics"
                }
            }
            
            # Store cache structure
            await self.set("cache_structure", cache_structure, 86400)  # 24 hours
            
            # Initialize performance metrics
            initial_metrics = {
                "cache_hits": 0,
                "cache_misses": 0,
                "total_requests": 0,
                "average_response_time": 0.0,
                "last_updated": datetime.now().isoformat()
            }
            await self.set("performance_metrics", initial_metrics, 300)
            
            return {
                "status": "success",
                "cache_structure": cache_structure,
                "initial_metrics": initial_metrics
            }
            
        except Exception as e:
            logger.error(f"Cache structure setup error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cache value"""
        if not self.connected:
            return False
        
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value, default=str)
            else:
                serialized = str(value)
            
            # Set with TTL
            self.redis_client.setex(key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if not self.connected:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data
            return None
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete cache value"""
        if not self.connected:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        if not self.connected:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}")
            return 0
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics"""
        if not self.connected:
            return {"status": "redis_not_available"}
        
        try:
            info = self.redis_client.info()
            
            return {
                "status": "connected",
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1),
                "uptime": info.get("uptime_in_seconds"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cache info error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def warm_cache(self) -> Dict[str, Any]:
        """Warm up cache with frequently accessed data"""
        if not self.connected:
            return {"status": "redis_not_available"}
        
        try:
            # Warm up cache with common data
            warm_data = {
                "system_config": {
                    "consciousness_enhancement": 1.618,
                    "quantum_precision": "float32",
                    "gpu_acceleration": True,
                    "cache_enabled": True
                },
                "algorithm_configs": {
                    "wallace_transform": {
                        "iterations": 1000,
                        "precision": 0.001,
                        "prime_aligned_level": 1.618
                    },
                    "quantum_annealing": {
                        "qubits": 10,
                        "iterations": 1000,
                        "temperature": 1.0
                    }
                },
                "performance_benchmarks": {
                    "baseline_accuracy": 0.4,
                    "target_accuracy": 0.8,
                    "current_accuracy": 0.42
                }
            }
            
            # Store warm data
            for key, value in warm_data.items():
                await self.set(key, value, 3600)  # 1 hour TTL
            
            return {
                "status": "success",
                "warmed_keys": list(warm_data.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cache warming error: {e}")
            return {"status": "error", "error": str(e)}

async def main():
    """Main function for Redis cache setup"""
    logger.info("🚀 Starting Redis Cache Setup...")
    
    # Initialize cache setup
    cache_setup = RedisCacheSetup()
    
    if not cache_setup.connected:
        logger.error("❌ Redis not available. Please install and start Redis:")
        logger.error("   macOS: brew install redis && brew services start redis")
        logger.error("   Ubuntu: sudo apt install redis-server && sudo systemctl start redis")
        logger.error("   Docker: docker run -d -p 6379:6379 redis:alpine")
        return
    
    # Set up cache structure
    logger.info("📊 Setting up cache structure...")
    structure_result = await cache_setup.setup_cache_structure()
    print(f"Cache structure setup: {structure_result['status']}")
    
    # Warm up cache
    logger.info("🔥 Warming up cache...")
    warm_result = await cache_setup.warm_cache()
    print(f"Cache warming: {warm_result['status']}")
    
    # Get cache info
    logger.info("📈 Getting cache information...")
    cache_info = await cache_setup.get_cache_info()
    
    print("\n" + "="*60)
    print("🏆 REDIS CACHE SETUP COMPLETE")
    print("="*60)
    print(f"Status: {cache_info['status']}")
    print(f"Redis Version: {cache_info.get('redis_version', 'Unknown')}")
    print(f"Memory Usage: {cache_info.get('used_memory', 'Unknown')}")
    print(f"Connected Clients: {cache_info.get('connected_clients', 0)}")
    print(f"Hit Rate: {cache_info.get('hit_rate', 0):.2%}")
    print(f"Uptime: {cache_info.get('uptime', 0)} seconds")
    
    # Test cache operations
    logger.info("🧪 Testing cache operations...")
    
    # Test set/get
    test_key = "test_cache_key"
    test_value = {"test": "data", "timestamp": datetime.now().isoformat()}
    
    await cache_setup.set(test_key, test_value, 60)
    retrieved_value = await cache_setup.get(test_key)
    
    if retrieved_value == test_value:
        print("✅ Cache set/get test passed")
    else:
        print("❌ Cache set/get test failed")
    
    # Test pattern clearing
    await cache_setup.clear_pattern("test_*")
    cleared_value = await cache_setup.get(test_key)
    
    if cleared_value is None:
        print("✅ Cache pattern clearing test passed")
    else:
        print("❌ Cache pattern clearing test failed")
    
    print("\n✅ Redis cache setup complete!")
    print("💡 Cache is ready for high-performance data access")

if __name__ == "__main__":
    asyncio.run(main())
