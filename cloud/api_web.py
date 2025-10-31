"""Web API endpoints for Ada Cloud infrastructure.

This module provides HTTP endpoints for direct access to Modal functions.
"""

import modal
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
import os

# Modal app setup - use consistent app name
app = modal.App("AdaCloud")

# Images
base_image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi",
    "pydantic",
    "aiohttp",
])

# FastAPI app
web_app = FastAPI(title="Ada Cloud API", version="1.0.0")

@app.function(image=base_image)
@modal.fastapi_endpoint(method="GET")
def api():
    """Main API entry point."""
    return web_app

# Standalone web server for local development
def run_local_server():
    """Run the API server locally without Modal."""
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000)

# Import Modal functions after deployment
def _get_modal_functions():
    """Get references to deployed Modal functions."""
    try:
        # These will be created after full deployment
        infer_fn = modal.Function.lookup("AdaCloud", "ada_infer")
        train_fn = modal.Function.lookup("AdaCloud", "ada_train")
        mission_fn = modal.Function.lookup("AdaCloud", "ada_mission")
        optimize_fn = modal.Function.lookup("AdaCloud", "ada_optimize")
        health_fn = modal.Function.lookup("AdaCloud", "health_check")
        
        return {
            "infer": infer_fn,
            "train": train_fn,
            "mission": mission_fn,
            "optimize": optimize_fn,
            "health": health_fn,
        }
    except Exception as e:
        return None

@web_app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Ada Cloud API",
        "version": "1.0.0",
        "status": "running",
        "functions": ["infer", "train", "mission", "optimize", "health"]
    }

@web_app.get("/status")
async def status():
    """Health check endpoint."""
    try:
        functions = _get_modal_functions()
        if functions and functions["health"]:
            result = functions["health"].remote()
            return result
        else:
            return {
                "status": "partially_operational",
                "message": "Core services running, specialized functions not yet connected"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@web_app.post("/infer")
async def inference(request: Dict[str, Any]):
    """Inference endpoint."""
    try:
        functions = _get_modal_functions()
        if functions and functions["infer"]:
            result = functions["infer"].remote(request)
            return result
        else:
            return {
                "success": False,
                "error": "Inference function not available"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/train")
async def training(request: Dict[str, Any]):
    """Training endpoint."""
    try:
        model = request.pop("model", "default")
        training_data = request
        
        functions = _get_modal_functions()
        if functions and functions["train"]:
            result = functions["train"].remote(model, training_data)
            return result
        else:
            return {
                "success": False,
                "error": "Training function not available"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/mission")
async def mission(request: Dict[str, Any]):
    """Mission execution endpoint."""
    try:
        goal = request.pop("goal")
        context = request.pop("context", None)
        
        functions = _get_modal_functions()
        if functions and functions["mission"]:
            result = functions["mission"].remote(goal, context)
            return result
        else:
            return {
                "success": False,
                "error": "Mission function not available"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/optimize")
async def optimize(request: Dict[str, Any]):
    """Optimization endpoint."""
    try:
        functions = _get_modal_functions()
        if functions and functions["optimize"]:
            result = functions["optimize"].remote(request)
            return result
        else:
            return {
                "success": False,
                "error": "Optimization function not available"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    run_local_server()
