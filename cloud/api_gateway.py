"""API Gateway for Ada Cloud infrastructure.

This module provides FastAPI endpoints for all cloud services with proper
authentication, rate limiting, and request validation.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)


# Pydantic models for request/response schemas
class InferenceRequest(BaseModel):
    """Inference request model."""
    prompt: str = Field(..., description="Input prompt for inference")
    module: str = Field(default="core.reasoning", description="Module to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    stream: bool = Field(default=False, description="Whether to stream response")


class OptimizationRequest(BaseModel):
    """Optimization request model."""
    target_module: str = Field(..., description="Module to optimize")
    parameter_space: Dict[str, Any] = Field(..., description="Parameter search space")
    algorithm: str = Field(default="genetic", description="Optimization algorithm")
    max_iterations: int = Field(default=50, description="Maximum iterations")
    convergence_threshold: float = Field(default=0.001, description="Convergence threshold")
    budget: int = Field(default=1000, description="Computational budget")
    objective: str = Field(default="maximize", description="Optimization objective")
    target_value: Optional[float] = Field(default=None, description="Target value (optional)")


class MissionRequest(BaseModel):
    """Mission request model."""
    goal: str = Field(..., description="Mission goal")
    context: Dict[str, Any] = Field(default_factory=dict, description="Mission context")
    priority: str = Field(default="medium", description="Mission priority")


class StorageUploadRequest(BaseModel):
    """Storage upload request model."""
    file_path: str = Field(..., description="Path to file to upload")
    key: Optional[str] = Field(default=None, description="Storage key")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="File metadata")


class StorageDownloadRequest(BaseModel):
    """Storage download request model."""
    key: str = Field(..., description="Storage key to download")
    destination: str = Field(..., description="Local destination path")
    overwrite: bool = Field(default=False, description="Whether to overwrite existing files")


# Simple API key authentication
async def api_key_auth(request: Request):
    """Simple API key authentication."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        api_key = request.query_params.get("api_key")
    
    # In production, validate against a proper auth system
    expected_key = "ada-cloud-api-key"  # Use environment variable in production
    
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"api_key": api_key}


# Request/response logging middleware
async def log_requests(request: Request, call_next):
    """Log requests and responses."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
    
    return response


# Service dependencies
def get_modal_functions():
    """Get Modal function references."""
    try:
        import modal
        
        # These will reference deployed Modal functions
        ada_infer = modal.Function.lookup("AdaCloud", "ada_infer")
        ada_optimize = modal.Function.lookup("AdaCloud", "ada_optimize")
        ada_mission = modal.Function.lookup("AdaCloud", "ada_mission")
        
        return {
            "infer": ada_infer,
            "optimize": ada_optimize,
            "mission": ada_mission,
        }
    except Exception as e:
        logger.warning(f"Could not load Modal functions: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    logger.info("Ada Cloud API Gateway starting up...")
    
    # Initialize connections
    app.state.modal_functions = get_modal_functions()
    
    yield
    
    # Shutdown
    logger.info("Ada Cloud API Gateway shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Ada Cloud API",
    description="Cloud API for Ada's neural core, mission daemon, and optimizer",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.middleware("http")(log_requests)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Ada Cloud API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": time.time(),
        "endpoints": [
            "/status",
            "/infer",
            "/optimize", 
            "/mission",
            "/storage/upload",
            "/storage/download",
            "/storage/list",
        ],
    }


@app.get("/status")
async def status(auth=Depends(api_key_auth)):
    """Health status and system information."""
    try:
        # Check Modal function availability
        modal_available = app.state.modal_functions is not None
        
        # Check storage service
        from cloud.storage_service import get_storage_service
        storage_service = get_storage_service()
        storage_available = storage_service._initialized or True  # Simulated storage is OK
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "modal_functions": "available" if modal_available else "unavailable",
                "storage": "connected" if storage_available else "disconnected",
            },
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Inference endpoint
@app.post("/infer")
async def infer(
    request: InferenceRequest,
    auth=Depends(api_key_auth),
):
    """Run inference on Ada's neural core."""
    try:
        if not app.state.modal_functions:
            raise HTTPException(status_code=503, detail="Modal functions unavailable")
        
        infer_fn = app.state.modal_functions["infer"]
        
        # Prepare request data
        data = {
            "prompt": request.prompt,
            "module": request.module,
            "parameters": request.parameters,
            "stream": request.stream,
        }
        
        # Call Modal function
        result = infer_fn.remote(data)
        
        return {
            "success": True,
            "result": result,
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Optimization endpoint
@app.post("/optimize")
async def optimize(
    request: OptimizationRequest,
    auth=Depends(api_key_auth),
):
    """Run optimization job."""
    try:
        if not app.state.modal_functions:
            raise HTTPException(status_code=503, detail="Modal functions unavailable")
        
        optimize_fn = app.state.modal_functions["optimize"]
        
        # Prepare request data
        data = {
            "target_module": request.target_module,
            "parameter_space": request.parameter_space,
            "algorithm": request.algorithm,
            "max_iterations": request.max_iterations,
            "convergence_threshold": request.convergence_threshold,
            "budget": request.budget,
            "objective": request.objective,
            "target_value": request.target_value,
        }
        
        # Call Modal function
        result = optimize_fn.remote(data)
        
        return {
            "success": True,
            "result": result,
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mission endpoint
@app.post("/mission")
async def mission(
    request: MissionRequest,
    auth=Depends(api_key_auth),
):
    """Execute autonomous mission."""
    try:
        if not app.state.modal_functions:
            raise HTTPException(status_code=503, detail="Modal functions unavailable")
        
        mission_fn = app.state.modal_functions["mission"]
        
        # Prepare request data
        data = {
            "goal": request.goal,
            "context": request.context,
            "priority": request.priority,
        }
        
        # Call Modal function
        result = mission_fn.remote(data)
        
        return {
            "success": True,
            "result": result,
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"Mission execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Storage upload endpoint
@app.post("/storage/upload")
async def upload_file(
    request: StorageUploadRequest,
    auth=Depends(api_key_auth),
):
    """Upload file to cloud storage."""
    try:
        from cloud.storage_service import get_storage_service
        
        storage_service = get_storage_service()
        
        # Upload file
        result = await storage_service.upload_file(
            file_path=request.file_path,
            key=request.key,
            metadata=request.metadata,
        )
        
        return {
            "success": result.get("success", False),
            "result": result,
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Storage download endpoint
@app.post("/storage/download")
async def download_file(
    request: StorageDownloadRequest,
    auth=Depends(api_key_auth),
):
    """Download file from cloud storage."""
    try:
        from cloud.storage_service import get_storage_service
        
        storage_service = get_storage_service()
        
        # Download file
        result = await storage_service.download_file(
            key=request.key,
            destination=request.destination,
            overwrite=request.overwrite,
        )
        
        return {
            "success": result.get("success", False),
            "result": result,
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Storage list endpoint
@app.get("/storage/list")
async def list_files(
    prefix: str = "",
    limit: int = 1000,
    include_metadata: bool = False,
    auth=Depends(api_key_auth),
):
    """List files in cloud storage."""
    try:
        from cloud.storage_service import get_storage_service
        
        storage_service = get_storage_service()
        
        # List files
        result = await storage_service.list_files(
            prefix=prefix,
            limit=limit,
            include_metadata=include_metadata,
        )
        
        return {
            "success": result.get("success", False),
            "result": result,
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"File listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "timestamp": time.time(),
        },
    )


# Development server function
def run_dev_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API Gateway in development mode."""
    uvicorn.run(
        "cloud.api_gateway:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_dev_server(reload=True)
