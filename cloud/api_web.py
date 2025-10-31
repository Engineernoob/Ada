"""Web API endpoints for Ada Cloud infrastructure.

This module provides HTTP endpoints for direct access to Modal functions.
"""

import modal
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
import os

# Modal app setup - use consistent app name
app = modal.App("AdaCloudFinal")

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
        try:
            infer_fn = modal.Function.lookup("AdaCloudFinal", "ada_infer")
            train_fn = modal.Function.lookup("AdaCloudFinal", "ada_train")
            mission_fn = modal.Function.lookup("AdaCloudFinal", "ada_mission")
            optimize_fn = modal.Function.lookup("AdaCloudFinal", "ada_optimize")
            # Note: health_check doesn't exist, use simple response
        except Exception:
            return None
        
        return {
            "infer": infer_fn,
            "train": train_fn,
            "mission": mission_fn,
            "optimize": optimize_fn,
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
        if functions:
            return {
                "status": "operational",
                "message": "All services running",
                "functions": list(functions.keys())
            }
        else:
            return {
                "status": "partially_operational",
                "message": "Web service running, specialized functions not yet connected"
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
        model = request.get("model", "default")
        # Pass the entire request as training_data
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

@web_app.post("/rate")
async def rate_feedback(request: Dict[str, Any]):
    """Feedback rating endpoint for adaptive learning.
    
    Logs user feedback for model improvement via reinforcement learning.
    """
    try:
        prompt = request.get("prompt")
        response = request.get("response")
        reward = request.get("reward")
        
        if not prompt or not response or reward is None:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: prompt, response, reward"
            )
        
        # Validate reward range
        reward_val = float(reward)
        if not -1.0 <= reward_val <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="Reward must be between -1.0 and 1.0"
            )
        
        # Log feedback directly (since this is a web endpoint)
        try:
            from neural.enhanced_policy_network import log_feedback
            log_feedback(prompt, response, reward_val)
            
            return {
                "success": True,
                "message": "Feedback recorded for adaptive learning",
                "reward": reward_val
            }
        except ImportError:
            # Fallback to simple file logging if module not available
            import json
            from datetime import datetime
            from pathlib import Path
            
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "training_feedback.jsonl"
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "reward": reward_val
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(entry) + "\n")
            
            return {
                "success": True,
                "message": "Feedback recorded",
                "reward": reward_val
            }
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid reward value: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    run_local_server()
