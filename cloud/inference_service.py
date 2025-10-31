"""Inference service for Ada Cloud infrastructure.

This module provides specialized inference capabilities for different Ada modules,
optimized for Modal's serverless environment with proper resource management.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Generator
import time
import json

import torch
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class AdaInferenceService:
    """High-performance inference service for Ada modules."""
    
    def __init__(self, storage_base_path: str = "/root/ada/storage"):
        """Initialize inference service.
        
        Args:
            storage_base_path: Base path for persistent storage
        """
        self.storage_base_path = Path(storage_base_path)
        self.models_cache = {}
        self.device = self._get_device()
        
        # Initialize module components
        self.reasoning_engine = None
        self._initialize_modules()
    
    def _get_device(self) -> torch.device:
        """Determine optimal device for inference."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps") 
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    def _initialize_modules(self):
        """Initialize all available inference modules."""
        try:
            # Initialize reasoning engine lazily
            logger.info("Initializing inference modules")
            
            # Check for available models
            models_dir = self.storage_base_path / "models"
            if models_dir.exists():
                available_models = list(models_dir.glob("*.pt"))
                logger.info(f"Found {len(available_models)} available models")
            
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
    
    def _load_reasoning_engine(self):
        """Load the reasoning engine module."""
        if self.reasoning_engine is None:
            try:
                # Import here to avoid heavy imports in cold starts
                from core import ReasoningEngine
                self.reasoning_engine = ReasoningEngine()
                logger.info("Reasoning engine loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load reasoning engine: {e}")
                raise
    
    def inference(
        self,
        prompt: str,
        module: str = "core.reasoning",
        parameters: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Run inference using specified module.
        
        Args:
            prompt: Input prompt for inference
            module: Module identifier for inference
            parameters: Additional inference parameters
            stream: Whether to stream response
            
        Returns:
            Inference result dictionary
        """
        parameters = parameters or {}
        start_time = time.time()
        
        try:
            if module == "core.reasoning":
                return self._reasoning_inference(prompt, parameters, stream)
            elif module == "core.autonomous_planner":
                return self._planning_inference(prompt, parameters, stream)
            elif module == "conversation":
                return self._conversation_inference(prompt, parameters, stream)
            elif module == "voice":
                return self._voice_inference(prompt, parameters, stream)
            else:
                raise ValueError(f"Unknown module: {module}")
                
        except Exception as e:
            logger.error(f"Inference failed for module {module}: {e}")
            return {
                "success": False,
                "error": str(e),
                "module": module,
                "execution_time": time.time() - start_time,
            }
    
    def _reasoning_inference(
        self,
        prompt: str,
        parameters: Dict[str, Any],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Run core reasoning inference."""
        try:
            # Load reasoning engine with error handling
            try:
                from core import ReasoningEngine
                engine = ReasoningEngine()
            except ImportError as e:
                logger.error(f"Failed to import ReasoningEngine: {e}")
                # Fallback response
                return {
                    "success": False,
                    "error": f"Reasoning engine not available: {e}",
                    "module": "core.reasoning",
                    "response": "I apologize, but the reasoning engine is currently unavailable. Please check the installation of required dependencies like torch and transformers.",
                }
            
            if stream:
                # Implement streaming for reasoning
                return self._stream_reasoning(prompt, parameters)
            else:
                # Standard inference
                max_tokens = parameters.get("max_tokens", 500)
                temperature = parameters.get("temperature", 0.7)
                
                try:
                    result = engine.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except Exception as gen_error:
                    logger.error(f"Generation failed: {gen_error}")
                    return {
                        "success": False,
                        "error": f"Generation failed: {gen_error}",
                        "module": "core.reasoning",
                        "response": "I apologize, but I encountered an error during text generation. This might be due to missing model files or configuration issues.",
                    }
                
                return {
                    "success": True,
                    "response": result.text,
                    "confidence": float(getattr(result, 'confidence', 0.8)),
                    "module": "core.reasoning",
                    "tokens_used": getattr(result, 'tokens_used', 0),
                    "generation_time": getattr(result, 'generation_time', 0.0),
                }
                
        except Exception as e:
            logger.error(f"Reasoning inference failed: {e}")
            raise
    
    def _planning_inference(
        self,
        prompt: str,
        parameters: Dict[str, Any],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Run autonomous planning inference."""
        try:
            try:
                from planner import IntentEngine, Planner
                intent_engine = IntentEngine()
                intents = intent_engine.infer([prompt])
            except ImportError as e:
                logger.error(f"Failed to import planning modules: {e}")
                return {
                    "success": False,
                    "error": f"Planning modules not available: {e}",
                    "module": "core.autonomous_planner",
                }
            except Exception as e:
                logger.error(f"Intent inference failed: {e}")
                return {
                    "success": False,
                    "error": f"Intent inference failed: {e}",
                    "module": "core.autonomous_planner",
                }
            
            if not intents:
                return {
                    "success": False,
                    "error": "Could not infer intent from prompt",
                    "module": "core.autonomous_planner",
                }
            
            planner = Planner()
            plan = planner.plan(intents[0])
            
            if not plan:
                return {
                    "success": False,
                    "error": "Could not generate plan",
                    "module": "core.autonomous_planner",
                }
            
            # Convert plan to actionable steps
            steps = []
            for step in plan.steps:
                steps.append({
                    "description": step.description,
                    "type": step.type if hasattr(step, 'type') else "action",
                    "tool": step.tool if hasattr(step, 'tool') else None,
                    "parameters": step.parameters if hasattr(step, 'parameters') else {},
                })
            
            return {
                "success": True,
                "response": f"Generated plan with {len(steps)} steps",
                "module": "core.autonomous_planner",
                "plan": {
                    "goal": plan.goal if hasattr(plan, 'goal') else "Executing plan",
                    "steps": steps,
                    "confidence": 0.85,
                },
                "intents": [{"text": intent.text, "confidence": intent.confidence} for intent in intents],
            }
            
        except Exception as e:
            logger.error(f"Planning inference failed: {e}")
            raise
    
    def _conversation_inference(
        self,
        prompt: str,
        parameters: Dict[str, Any],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Run conversation inference with memory."""
        try:
            try:
                from core.memory import ConversationStore
                # Load conversation store
                store = ConversationStore()
                # Add user message
                user_message = store.add_message("user", prompt)
            except ImportError as e:
                logger.error(f"Failed to import ConversationStore: {e}")
                # Fallback to regular reasoning without conversation context
                return self._reasoning_inference(prompt, parameters, stream)
            except Exception as e:
                logger.error(f"Conversation store error: {e}")
                # Fallback to regular reasoning without conversation context
                return self._reasoning_inference(prompt, parameters, stream)
            
            # Generate response (fallback to regular reasoning)
            reasoning_result = self._reasoning_inference(prompt, parameters, stream)
            
            if reasoning_result["success"]:
                # Add assistant response
                assistant_message = store.add_message("assistant", reasoning_result["response"])
                
                # Update response with conversation context
                reasoning_result["conversation_id"] = store.conversation.id
                reasoning_result["message_id"] = assistant_message.id
                
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Conversation inference failed: {e}")
            raise
    
    def _voice_inference(
        self,
        prompt: str,
        parameters: Dict[str, Any],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Handle voice-related inference (TTS/STT preprocessing)."""
        try:
            # This is primarily for voice command processing
            # The actual TTS/STT is handled elsewhere
            
            # Check if prompt contains voice commands
            voice_commands = ["speak", "say", "talk", "listen", "whisper"]
            is_voice_command = any(cmd in prompt.lower() for cmd in voice_commands)
            
            if is_voice_command:
                # Process as voice command
                response = self._reasoning_inference(prompt, parameters, stream)
                response["voice_activated"] = True
                response["requires_tts"] = True
                return response
            else:
                # Regular text processing
                return self._reasoning_inference(prompt, parameters, stream)
                
        except Exception as e:
            logger.error(f"Voice inference failed: {e}")
            raise
    
    def _stream_reasoning(
        self,
        prompt: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stream reasoning response (simplified for Modal)."""
        # For Modal, we'll return a single response with streaming metadata
        # In a real implementation, this would use proper streaming
        result = self._reasoning_inference(prompt, parameters, stream=False)
        result["streaming"] = True
        result["protocol"] = "single-token"  # Placeholder
        return result
    
    def batch_inference(
        self,
        prompts: List[str],
        module: str = "core.reasoning",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run inference on multiple prompts in batch."""
        parameters = parameters or {}
        results = []
        
        logger.info(f"Running batch inference on {len(prompts)} prompts")
        
        for i, prompt in enumerate(prompts):
            try:
                result = self.inference(prompt, module, parameters)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Batch inference failed for prompt {i}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "module": module,
                    "batch_index": i,
                })
        
        return results
    
    def get_model_info(self, module: str) -> Dict[str, Any]:
        """Get information about loaded model for specified module."""
        try:
            return {
                "module": module,
                "device": str(self.device),
                "model_loaded": True if module == "core.reasoning" and self.reasoning_engine else False,
                "model_size": "unknown",  # TODO: Calculate actual model size
                "supported_features": ["text_generation", "reasoning", "planning"],
                "memory_usage": self._get_memory_usage(),
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            if self.device.type == "cuda":
                allocated = torch.cuda.memory_allocated(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                return {
                    "allocated_mb": allocated / 1024**2,
                    "reserved_mb": reserved / 1024**2,
                }
            else:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                return {
                    "allocated_mb": memory_info.rss / 1024**2,
                    "system_memory_percent": process.memory_percent(),
                }
        except Exception:
            return {"error": "Could not measure memory usage"}


# Modal function wrapper for inference
def ada_infer(data: Dict[str, Any]) -> Dict[str, Any]:
    """Modal inference function wrapper."""
    try:
        service = AdaInferenceService()
        
        prompt = data.get("prompt", "")
        module = data.get("module", "core.reasoning")
        parameters = data.get("parameters", {})
        stream = data.get("stream", False)
        
        return service.inference(prompt, module, parameters, stream)
        
    except Exception as e:
        logger.error(f"Modal inference failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "module": data.get("module", "core.reasoning"),
        }


if __name__ == "__main__":
    # Local testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Ada Inference Service")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--module", default="core.reasoning", help="Module to use")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    
    args = parser.parse_args()
    
    # Test inference
    service = AdaInferenceService()
    result = service.inference(
        prompt=args.prompt,
        module=args.module,
        parameters={
            "max_tokens": args.max_tokens,
            "temperature": args.temperature
        }
    )
    
    print(json.dumps(result, indent=2))
