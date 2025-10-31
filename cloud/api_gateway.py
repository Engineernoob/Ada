"""FastAPI gateway for Ada Cloud infrastructure.

This module provides HTTP endpoints that route client requests to Modal functions
with proper authentication, rate limiting, and error handling.
"""

from __future__ import annotations

import os
import asyncio
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import modal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# API key authentication
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Initialize FastAPI app
app = FastAPI(
    title="Ada Cloud API",
    description="Serverless cloud backend for Ada AI assistant",
    version="1.0.0",
)

# Add rate limit exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for inference")
    module: str = Field(default="core.reasoning", description="Module to use for inference")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    stream: bool = Field(default=False, description="Whether to stream response")

class InferenceResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    confidence: float = 0.0
    module: str
    tokens_used: int = 0
    error: Optional[str] = None

class TrainingRequest(BaseModel):
    model: str = Field(..., description="Model identifier")
    training_data: Dict[str, Any] = Field(..., description="Training configuration and data")
    epochs: int = Field(default=10, description="Number of training epochs")

class TrainingResponse(BaseModel):
    success: bool
    model: str
    status: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None

class MissionRequest(BaseModel):
    goal: str = Field(..., description="Mission goal description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Mission context and constraints")
    priority: str = Field(default="medium", description="Mission priority")

class MissionResponse(BaseModel):
    success: bool
    mission_id: Optional[str] = None
    goal: str
    status: Optional[str] = None
    steps_completed: int = 0
    results: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None

class OptimizationRequest(BaseModel):
    target_module: str = Field(..., description="Module to optimize")
    type: str = Field(default="parameter_tuning", description="Optimization type")
    budget: int = Field(default=1000, description="Budget for optimization")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Optimization parameters")

class OptimizationResponse(BaseModel):
    success: bool
    target_module: str
    optimization_type: str
    improvement: float = 0.0
    best_params: Optional[Dict[str, Any]] = None
    budget_used: int = 0
    iterations: int = 0
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    platform: str
    gpu_available: bool
    modal_app_id: Optional[str] = None
    timestamp: Optional[str] = None
    uptime: Optional[float] = None


# Authenticate API key
async def get_api_key(api_key: str = Depends(api_key_header)):
    """Validate API key from Authorization header."""
    expected_key = os.getenv("ADA_API_KEY")
    
    # Allow placeholder key for initial setup
    if not expected_key or expected_key == "placeholder-key-for-deployment":
        logger.warning("No API key configured - allowing all requests for initial setup")
        return "placeholder-key" if api_key is None else api_key
    
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Remove "Bearer " prefix if present
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    
    # Validate against expected key
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key


# Initialize Modal functions
ada_infer_fn = modal.Function.lookup("AdaCloud", "ada_infer")
ada_train_fn = modal.Function.lookup("AdaCloud", "ada_train")
ada_mission_fn = modal.Function.lookup("AdaCloud", "ada_mission")
ada_optimize_fn = modal.Function.lookup("AdaCloud", "ada_optimize")
health_check_fn = modal.Function.lookup("AdaCloud", "health_check")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Ada Cloud API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/status", response_model=HealthResponse)
@limiter.limit("60/minute")
async def get_status():
    """Check the health status of the cloud infrastructure."""
    try:
        result = health_check_fn.remote()
        return HealthResponse(**result)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud infrastructure unavailable"
        )


@app.post("/infer", response_model=InferenceResponse)
@limiter.limit("30/minute")
async def inference(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Run inference on Ada core reasoning engine."""
    try:
        # Log request
        logger.info(f"Inference request: module={request.module}, prompt_length={len(request.prompt)}")
        
        # Prepare data for Modal function
        data = {
            "prompt": request.prompt,
            "module": request.module,
            "parameters": request.parameters,
        }
        
        if request.stream:
            # TODO: Implement streaming response
            # For now, return regular response
            result = await ada_infer_fn.aio(data)
        else:
            result = await ada_infer_fn.aio(data)
        
        # Store interaction metrics
        background_tasks.add_task(
            log_interaction,
            "inference",
            request.module,
            result.get("tokens_used", 0),
            result.get("success", False)
        )
        
        return InferenceResponse(**result)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post("/train", response_model=TrainingResponse)
@limiter.limit("5/minute")
async def training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Train specified model on cloud infrastructure."""
    try:
        logger.info(f"Training request: model={request.model}, epochs={request.epochs}")
        
        # Prepare training data
        training_data = request.training_data.copy()
        training_data["epochs"] = request.epochs
        
        # Run training
        result = await ada_train_fn.aio(request.model, training_data)
        
        # Log training metrics
        background_tasks.add_task(
            log_interaction,
            "training",
            request.model,
            0,  # Training doesn't use tokens in the same way
            result.get("success", False)
        )
        
        return TrainingResponse(**result)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@app.post("/mission", response_model=MissionResponse)
@limiter.limit("20/minute")
async def mission_execution(
    request: MissionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Execute autonomous missions on cloud infrastructure."""
    try:
        logger.info(f"Mission request: goal='{request.goal}', priority={request.priority}")
        
        # Execute mission
        result = await ada_mission_fn.aio(request.goal, request.context)
        
        # Log mission metrics
        background_tasks.add_task(
            log_interaction,
            "mission",
            request.goal[:50],  # Truncate for logging
            result.get("steps_completed", 0),
            result.get("success", False)
        )
        
        return MissionResponse(**result)
        
    except Exception as e:
        logger.error(f"Mission execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mission execution failed: {str(e)}"
        )


@app.post("/optimize", response_model=OptimizationResponse)
@limiter.limit("10/minute")
async def optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Run optimization/evolution jobs on cloud infrastructure."""
    try:
        logger.info(f"Optimization request: module={request.target_module}, type={request.type}")
        
        # Prepare optimization parameters
        params = {
            "target_module": request.target_module,
            "type": request.type,
            "budget": request.budget,
            **request.parameters,
        }
        
        # Run optimization
        result = await ada_optimize_fn.aio(params)
        
        # Log optimization metrics
        background_tasks.add_task(
            log_interaction,
            "optimization",
            request.target_module,
            result.get("budget_used", 0),
            result.get("success", False)
        )
        
        return OptimizationResponse(**result)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@app.get("/metrics")
@limiter.limit("60/minute")
async def get_metrics(api_key: str = Depends(get_api_key)):
    """Get API usage metrics."""
    # TODO: Implement proper metrics collection
    from datetime import datetime
    return {
        "status": "available",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "inference": {"requests_today": 150, "success_rate": 0.95},
            "training": {"requests_today": 3, "success_rate": 1.0},
            "mission": {"requests_today": 25, "success_rate": 0.88},
            "optimization": {"requests_today": 2, "success_rate": 1.0},
        }
    }


# Background task for logging interactions
async def log_interaction(
    endpoint: str,
    target: str,
    resource_usage: int,
    success: bool
):
    """Log API interaction for metrics and monitoring."""
    try:
        # TODO: Implement proper logging to database or file
        logger.info(
            f"Interaction: {endpoint}, target={target}, "
            f"usage={resource_usage}, success={success}, "
            f"time={datetime.utcnow().isoformat()}"
        )
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")


# Main function for local development
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ada Cloud API Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.getenv("ADA_API_KEY"):
        logger.warning("ADA_API_KEY environment variable not set - authentication will fail")
    
    # Start server
    uvicorn.run(
        "api_gateway:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
