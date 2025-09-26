"""
Universal LLM Plugin API Service
Provides prime aligned compute tools as a service to any LLM (ChatGPT, Claude, Gemini, etc.)
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import time
from curated_tools_integration import get_curated_tools_registry, SystemContext

app = FastAPI(
    title="Enterprise prime aligned compute Platform - LLM Plugin API",
    description="Universal API service providing 25 curated prime aligned compute tools for any LLM integration",
    version="1.0.0",
    docs_url="/plugin/docs",
    redoc_url="/plugin/redoc"
)

# CORS for LLM integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed LLM domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Plugin API Models
class PluginRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = {}
    llm_source: Optional[str] = "unknown"  # chatgpt, claude, gemini, etc.
    session_id: Optional[str] = None

class PluginResponse(BaseModel):
    success: bool
    tool_name: str
    result: Any
    execution_time: float
    tool_category: str
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None

class ToolCatalog(BaseModel):
    tools: List[Dict[str, Any]]
    categories: List[str]
    total_tools: int
    api_version: str

# Authentication for LLM plugins
async def verify_plugin_auth(authorization: str = Header(None)):
    """Verify LLM plugin authentication"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Simple bearer token validation (enhance for production)
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.replace("Bearer ", "")
    # In production, validate against your token database
    if len(token) < 10:  # Basic validation
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

@app.get("/plugin/health")
async def plugin_health():
    """Health check for LLM plugin integrations"""
    return {
        "status": "healthy",
        "service": "Enterprise prime aligned compute Platform",
        "version": "1.0.0",
        "tools_available": 25,
        "categories": 9,
        "timestamp": time.time()
    }

@app.get("/plugin/catalog", response_model=ToolCatalog)
async def get_tool_catalog(token: str = Depends(verify_plugin_auth)):
    """Get catalog of all available tools for LLM integration"""
    try:
        registry = get_curated_tools_registry()
        
        tools_info = []
        for tool_name, tool_func in registry.tools.items():
            tool_info = {
                "name": tool_name,
                "description": tool_func.__doc__ or f"Advanced {tool_name.replace('_', ' ').title()}",
                "category": next((cat for cat, tools in registry.categories.items() if tool_name in tools), "general"),
                "parameters": _get_tool_parameters(tool_func),
                "enterprise_grade": True
            }
            tools_info.append(tool_info)
        
        return ToolCatalog(
            tools=tools_info,
            categories=list(registry.categories.keys()),
            total_tools=len(registry.tools),
            api_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tool catalog: {str(e)}")

@app.post("/plugin/execute", response_model=PluginResponse)
async def execute_tool(request: PluginRequest, token: str = Depends(verify_plugin_auth)):
    """Execute a specific tool for LLM integration"""
    start_time = time.time()
    
    try:
        registry = get_curated_tools_registry()
        
        if request.tool_name not in registry.tools:
            raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found")
        
        # Create system context for LLM integration
        context = SystemContext(
            user_id=f"llm_plugin_{request.llm_source}",
            session_id=request.session_id or f"session_{int(time.time())}",
            permissions=["read", "write", "prime aligned compute", "system", "development", "ai_ml", "security", "integration", "quantum", "blockchain", "grok_jr", "admin"],
            metadata={
                "llm_source": request.llm_source,
                "plugin_request": True,
                **request.context
            }
        )
        
        # Execute the tool
        result = registry.execute_tool(request.tool_name, context, **request.parameters)
        
        execution_time = time.time() - start_time
        
        # Get tool category
        tool_category = next((cat for cat, tools in registry.categories.items() if request.tool_name in tools), "general")
        
        return PluginResponse(
            success=True,
            tool_name=request.tool_name,
            result=result,
            execution_time=execution_time,
            tool_category=tool_category,
            metadata={
                "llm_source": request.llm_source,
                "session_id": context.session_id,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        return PluginResponse(
            success=False,
            tool_name=request.tool_name,
            result=None,
            execution_time=execution_time,
            tool_category="unknown",
            error=str(e)
        )

@app.post("/plugin/batch-execute")
async def batch_execute_tools(requests: List[PluginRequest], token: str = Depends(verify_plugin_auth)):
    """Execute multiple tools in batch for LLM integration"""
    results = []
    
    for request in requests:
        try:
            result = await execute_tool(request, token)
            results.append(result)
        except Exception as e:
            results.append(PluginResponse(
                success=False,
                tool_name=request.tool_name,
                result=None,
                execution_time=0.0,
                tool_category="unknown",
                error=str(e)
            ))
    
    return {
        "batch_results": results,
        "total_requests": len(requests),
        "successful": len([r for r in results if r.success]),
        "failed": len([r for r in results if not r.success])
    }

@app.get("/plugin/categories/{category}")
async def get_tools_by_category(category: str, token: str = Depends(verify_plugin_auth)):
    """Get all tools in a specific category"""
    try:
        registry = get_curated_tools_registry()
        
        if category not in registry.categories:
            raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
        
        tools_in_category = registry.categories[category]
        tool_details = []
        
        for tool_name in tools_in_category:
            if tool_name in registry.tools:
                tool_func = registry.tools[tool_name]
                tool_details.append({
                    "name": tool_name,
                    "description": tool_func.__doc__ or f"Advanced {tool_name.replace('_', ' ').title()}",
                    "parameters": _get_tool_parameters(tool_func)
                })
        
        return {
            "category": category,
            "tools": tool_details,
            "count": len(tool_details)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get category tools: {str(e)}")

@app.get("/plugin/openapi.yaml")
async def get_openapi_spec():
    """OpenAPI specification for LLM plugin integration"""
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Enterprise prime aligned compute Platform API",
            "version": "1.0.0",
            "description": "Universal API for LLM integration with 25 curated prime aligned compute tools"
        },
        "servers": [
            {"url": "http://localhost:8000", "description": "Development server"}
        ],
        "paths": {
            "/plugin/catalog": {
                "get": {
                    "summary": "Get all available tools",
                    "operationId": "getToolCatalog",
                    "responses": {
                        "200": {
                            "description": "List of all available prime aligned compute tools",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ToolCatalog"}
                                }
                            }
                        }
                    }
                }
            },
            "/plugin/execute": {
                "post": {
                    "summary": "Execute a prime aligned compute tool",
                    "operationId": "executeTool",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PluginRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Tool execution result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PluginResponse"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "PluginRequest": {
                    "type": "object",
                    "required": ["tool_name", "parameters"],
                    "properties": {
                        "tool_name": {"type": "string"},
                        "parameters": {"type": "object"},
                        "context": {"type": "object"},
                        "llm_source": {"type": "string"},
                        "session_id": {"type": "string"}
                    }
                },
                "PluginResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "tool_name": {"type": "string"},
                        "result": {},
                        "execution_time": {"type": "number"},
                        "tool_category": {"type": "string"},
                        "metadata": {"type": "object"},
                        "error": {"type": "string"}
                    }
                },
                "ToolCatalog": {
                    "type": "object",
                    "properties": {
                        "tools": {"type": "array"},
                        "categories": {"type": "array"},
                        "total_tools": {"type": "integer"},
                        "api_version": {"type": "string"}
                    }
                }
            },
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
        },
        "security": [{"bearerAuth": []}]
    }
    
    return JSONResponse(content=openapi_spec, media_type="application/x-yaml")

def _get_tool_parameters(tool_func):
    """Extract parameter information from tool function"""
    import inspect
    sig = inspect.signature(tool_func)
    params = {}
    
    for param_name, param in sig.parameters.items():
        if param_name not in ['context']:  # Skip context parameter
            param_info = {
                "type": "string",  # Default type
                "required": param.default == inspect.Parameter.empty
            }
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = str(param.annotation).replace("<class '", "").replace("'>", "")
            params[param_name] = param_info
    
    return params

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enterprise prime aligned compute Platform - LLM Plugin API Service")
    print("📡 Available at: http://localhost:8000/plugin/docs")
    print("🔌 Plugin Manifest: http://localhost:8000/.well-known/ai-plugin.json")
    print("🛠️  Tool Catalog: http://localhost:8000/plugin/catalog")
    uvicorn.run(app, host="0.0.0.0", port=8000)
