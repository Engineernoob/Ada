"""Remote client for communicating with Ada Cloud infrastructure.

This module provides a client interface for connecting to Ada's cloud backend
with proper error handling, retries, and support for streaming responses.
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
import json
import time
from datetime import datetime, timedelta

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CloudConfig(BaseModel):
    """Configuration for cloud client."""
    endpoint: str = Field(..., description="Cloud API endpoint URL")
    api_key: str = Field(..., description="API authentication key")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Initial retry delay in seconds")
    enable_streaming: bool = Field(default=True, description="Enable response streaming")


class AdaCloudClient:
    """Async client for Ada Cloud infrastructure."""
    
    def __init__(self, config: Optional[CloudConfig] = None):
        """Initialize cloud client.
        
        Args:
            config: Cloud configuration (loads from environment if not provided)
        """
        if config is None:
            config = self._load_config()
        
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = config.endpoint.rstrip('/')
        
        # Performance metrics
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_latency = 0.0
    
    def _load_config(self) -> CloudConfig:
        """Load configuration from environment variables."""
        try:
            endpoint = os.getenv("ADA_CLOUD_ENDPOINT", "https://ada-cloud.modal.run")
            api_key = os.getenv("ADA_API_KEY")
            
            # Allow deployment without API key for initial setup
            if not api_key:
                logger.warning("ADA_API_KEY environment variable not set - using placeholder")
                api_key = "placeholder-key-for-deployment"
            
            return CloudConfig(
                endpoint=endpoint,
                api_key=api_key,
                timeout=int(os.getenv("ADA_CLOUD_TIMEOUT", "300")),
                max_retries=int(os.getenv("ADA_CLOUD_MAX_RETRIES", "3")),
                retry_delay=float(os.getenv("ADA_CLOUD_RETRY_DELAY", "1.0")),
                enable_streaming=os.getenv("ADA_CLOUD_ENABLE_STREAMING", "true").lower() == "true",
            )
            
        except Exception as e:
            logger.error(f"Failed to load cloud config: {e}")
            raise ValueError(f"Invalid cloud configuration: {e}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "AdaCloudClient/1.0",
            }
            
            # Set up SSL context (bypass verification for local testing)
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=connector,
            )
        
        return self._session
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Make HTTP request with retries and error handling."""
        session = await self._get_session()
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.request_count += 1
                
                # Skip auth header for placeholder key during initial setup
                headers = {}
                if self.config.api_key != "placeholder-key-for-deployment":
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                
                async with session.request(method, url, json=data, headers=headers) as response:
                    if stream and response.headers.get("content-type", "").startswith("text/event-stream"):
                        # Handle streaming response
                        async def stream_generator():
                            async for line in response.content:
                                if line:
                                    try:
                                        yield json.loads(line.decode().strip().lstrip("data: "))
                                    except json.JSONDecodeError:
                                        if line.strip() != "":  # Skip empty lines
                                            continue
                        return stream_generator()
                    
                    else:
                        # Handle regular response
                        response_data = await response.json()
                        
                        if response.status >= 400:
                            error_msg = response_data.get("error", f"HTTP {response.status}")
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=error_msg
                            )
                        
                        success = response_data.get("success", True)
                        if success:
                            self.success_count += 1
                        else:
                            self.error_count += 1
                        
                        latency = time.time() - start_time
                        self.total_latency += latency
                        
                        return response_data
            
            except Exception as e:
                last_error = e
                self.error_count += 1
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}")
                
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    break
        
        # All attempts failed
        logger.error(f"Request failed after {self.config.max_retries + 1} attempts: {last_error}")
        if isinstance(last_error, aiohttp.ClientResponseError):
            raise
        elif isinstance(last_error, asyncio.TimeoutError):
            raise aiohttp.ServerTimeoutError(f"Request timeout after {self.config.timeout}s")
        else:
            raise aiohttp.ClientError(f"Request failed: {last_error}")
    
    async def infer(
        self,
        module: str,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Run inference on specified module.
        
        Args:
            module: Module identifier (e.g., 'core.reasoning')
            prompt: Input prompt for inference
            parameters: Additional inference parameters
            stream: Whether to stream response
            
        Returns:
            Inference result or stream of results
        """
        data = {
            "prompt": prompt,
            "module": module,
            "parameters": parameters or {},
            "stream": stream and self.config.enable_streaming,
        }
        
        return await self._make_request("POST", "/infer", data, stream)
    
    async def train(
        self,
        model: str,
        training_data: Dict[str, Any],
        epochs: int = 10,
    ) -> Dict[str, Any]:
        """Train specified model.
        
        Args:
            model: Model identifier
            training_data: Training configuration and data
            epochs: Number of training epochs
            
        Returns:
            Training results
        """
        data = {
            "model": model,
            "training_data": training_data,
            "epochs": epochs,
        }
        
        return await self._make_request("POST", "/train", data)
    
    async def mission(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "medium",
    ) -> Dict[str, Any]:
        """Execute autonomous mission.
        
        Args:
            goal: Mission goal description
            context: Mission context and constraints
            priority: Mission priority
            
        Returns:
            Mission execution results
        """
        data = {
            "goal": goal,
            "context": context or {},
            "priority": priority,
        }
        
        return await self._make_request("POST", "/mission", data)
    
    async def optimize(
        self,
        target_module: str,
        optimization_type: str = "parameter_tuning",
        budget: int = 1000,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run optimization on target module.
        
        Args:
            target_module: Module to optimize
            optimization_type: Type of optimization
            budget: Budget for optimization
            parameters: Additional optimization parameters
            
        Returns:
            Optimization results
        """
        data = {
            "target_module": target_module,
            "type": optimization_type,
            "budget": budget,
            "parameters": parameters or {},
        }
        
        return await self._make_request("POST", "/optimize", data)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get cloud infrastructure status."""
        return await self._make_request("GET", "/status")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get API usage metrics."""
        return await self._make_request("GET", "/metrics")
    
    def get_client_metrics(self) -> Dict[str, Any]:
        """Get client-side performance metrics."""
        avg_latency = self.total_latency / max(self.request_count, 1)
        success_rate = self.success_count / max(self.request_count, 1)
        
        return {
            "requests": self.request_count,
            "successful": self.success_count,
            "errors": self.error_count,
            "success_rate": success_rate,
            "average_latency": avg_latency,
            "total_latency": self.total_latency,
        }
    
    async def health_check(self) -> bool:
        """Check if cloud endpoint is accessible."""
        try:
            status = await self.get_status()
            return status.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test full connection to cloud infrastructure."""
        results = {
            "endpoint": self._base_url,
            "authenticated": False,
            "status_check": False,
            "metrics_available": False,
            "sample_inference": False,
            "overall": False,
        }
        
        try:
            # Test basic status endpoint
            status = await self.get_status()
            results["status_check"] = True
            results["authenticated"] = True
            
            # Test metrics endpoint
            metrics = await self.get_metrics()
            results["metrics_available"] = True
            
            # Test simple inference
            inference_result = await self.infer(
                module="core.reasoning",
                prompt="Hello, this is a connection test.",
                parameters={"max_tokens": 10}
            )
            results["sample_inference"] = inference_result.get("success", False)
            
            # Overall success if all checks pass
            results["overall"] = all([
                results["authenticated"],
                results["status_check"],
                results["metrics_available"],
                results["sample_inference"],
            ])
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def close(self):
        """Close HTTP session and cleanup."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience functions for common operations
class AdaCloudClientSync:
    """Synchronous wrapper for Ada Cloud Client."""
    
    def __init__(self, config: Optional[CloudConfig] = None):
        self.config = config
        self._client: Optional[AdaCloudClient] = None
    
    def get_client(self) -> AdaCloudClient:
        """Get async client (creates if needed)."""
        if self._client is None:
            self._client = AdaCloudClient(self.config)
        return self._client
    
    def infer(
        self,
        module: str,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Synchronous inference wrapper."""
        async def _infer():
            client = self.get_client()
            async with client:
                return await client.infer(module, prompt, parameters, stream)
        
        return asyncio.run(_infer())
    
    def train(
        self,
        model: str,
        training_data: Dict[str, Any],
        epochs: int = 10,
    ) -> Dict[str, Any]:
        """Synchronous training wrapper."""
        async def _train():
            client = self.get_client()
            async with client:
                return await client.train(model, training_data, epochs)
        
        return asyncio.run(_train())
    
    def mission(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "medium",
    ) -> Dict[str, Any]:
        """Synchronous mission wrapper."""
        async def _mission():
            client = self.get_client()
            async with client:
                return await client.mission(goal, context, priority)
        
        return asyncio.run(_mission())
    
    def optimize(
        self,
        target_module: str,
        optimization_type: str = "parameter_tuning",
        budget: int = 1000,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synchronous optimization wrapper."""
        async def _optimize():
            client = self.get_client()
            async with client:
                return await client.optimize(target_module, optimization_type, budget, parameters)
        
        return asyncio.run(_optimize())
    
    def test_connection(self) -> Dict[str, Any]:
        """Synchronous connection test wrapper."""
        async def _test():
            client = self.get_client()
            async with client:
                return await client.test_connection()
        
        return asyncio.run(_test())


# Command line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ada Cloud Client Test")
    parser.add_argument("--action", default="test", choices=["test", "infer", "status", "metrics"])
    parser.add_argument("--endpoint", help="Cloud endpoint URL")
    parser.add_argument("--api-key", help="API authentication key")
    parser.add_argument("--module", default="core.reasoning", help="Module for inference")
    parser.add_argument("--prompt", help="Prompt for inference")
    
    args = parser.parse_args()
    
    # Create config from command line args if provided
    config = None
    if args.endpoint and args.api_key:
        config = CloudConfig(endpoint=args.endpoint, api_key=args.api_key)
    
    # Create client and run action
    if args.action == "test":
        client = AdaCloudClientSync(config)
        results = client.test_connection()
        print("Connection Test Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    
    elif args.action == "infer":
        if not args.prompt:
            print("Error: --prompt required for inference action")
            exit(1)
        
        client = AdaCloudClientSync(config)
        result = client.infer(args.module, args.prompt)
        print("Inference Results:")
        print(json.dumps(result, indent=2))
    
    elif args.action == "status":
        client = AdaCloudClientSync(config)
        status = client.get_client().get_status()
        asyncio.run(status)
        
    elif args.action == "metrics":
        client = AdaCloudClientSync(config)
        metrics = client.get_client().get_metrics()
        asyncio.run(metrics)
