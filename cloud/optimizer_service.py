"""Optimizer service for Ada Cloud infrastructure.

This module provides parameter optimization, model evolution, and
performance tuning capabilities optimized for Modal's serverless environment.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
import time
import random
import json
import numpy as np
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization available."""
    PARAMETER_TUNING = "parameter_tuning"
    HYPERPARAMETER_OPT = "hyperparameter_opt"
    MODEL_EVOLUTION = "model_evolution"
    ARCHITECTURE_SEARCH = "architecture_search"
    PERFORMANCE_TUNING = "performance_tuning"


class OptimizationStrategy(Enum):
    """Optimization strategies."""
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    GRID = "grid"
    GRADIENT = "gradient"


@dataclass
class OptimizationParameters:
    """Parameters for optimization run."""
    target_module: str
    optimization_type: OptimizationType
    strategy: OptimizationStrategy
    budget: int = 1000
    max_iterations: int = 100
    parameters: Dict[str, Any] = None
    constraints: Dict[str, Any] = None
    evaluation_metric: str = "accuracy"
    parameter_space: Dict[str, List[Any]] = None


class AdaOptimizerService:
    """Optimization service for Ada Cloud."""
    
    def __init__(self, storage_base_path: str = "/root/ada/storage"):
        """Initialize optimizer service.
        
        Args:
            storage_base_path: Base path for persistent storage
        """
        self.storage_base_path = storage_base_path
        self.active_optimizations = {}
        self.optimization_history = {}
        
        # Initialize optimizer system
        self._initialize_optimization_system()
    
    def _initialize_optimization_system(self):
        """Initialize the optimization system."""
        try:
            from optimizer import OptimizerSettings
            
            self.optimizer_settings = OptimizerSettings.from_settings()
            logger.info(f"Optimizer initialized with settings: enabled={self.optimizer_settings.enabled}")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            # Use default settings
            self.optimizer_settings = type('OptimizerSettings', (), {
                'enabled': True,
                'evaluation_interval_hours': 6,
                'max_population': 5,
                'mutation_rate': 0.15,
                'selection_top_k': 2,
                'preserve_best': True,
                'rollback_safe': True,
            })()
    
    def create_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new optimization job.
        
        Args:
            params: Optimization parameters and configuration
            
        Returns:
            Optimization creation result
        """
        try:
            optimization_id = str(uuid.uuid4())
            
            # Parse parameters
            opt_params = OptimizationParameters(
                target_module=params.get("target_module", "core"),
                optimization_type=OptimizationType(params.get("type", "parameter_tuning")),
                strategy=OptimizationStrategy(params.get("strategy", "random_search")),
                budget=params.get("budget", 1000),
                max_iterations=params.get("max_iterations", 100),
                parameters=params.get("parameters", {}),
                constraints=params.get("constraints", {}),
                evaluation_metric=params.get("evaluation_metric", "accuracy"),
                parameter_space=params.get("parameter_space", {}),
            )
            
            # Validate target module
            if not self._validate_target_module(opt_params.target_module):
                return {
                    "success": False,
                    "error": f"Invalid target module: {opt_params.target_module}",
                }
            
            # Store optimization
            self.active_optimizations[optimization_id] = {
                "id": optimization_id,
                "parameters": opt_params,
                "status": "pending",
                "created_at": time.time(),
                "current_iteration": 0,
                "best_score": float('-inf') if opt_params.evaluation_metric != "loss" else float('inf'),
                "best_params": {},
                "candidate_params": [],
                "evaluation_history": [],
                "budget_used": 0,
            }
            
            logger.info(f"Created optimization {optimization_id} for {opt_params.target_module}")
            
            return {
                "success": True,
                "optimization_id": optimization_id,
                "target_module": opt_params.target_module,
                "type": opt_params.optimization_type.value,
                "strategy": opt_params.strategy.value,
                "budget": opt_params.budget,
                "status": "pending",
            }
            
        except Exception as e:
            logger.error(f"Failed to create optimization: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def run_optimization(self, optimization_id: str, synchronous: bool = False) -> Dict[str, Any]:
        """Run an optimization job.
        
        Args:
            optimization_id: Optimization identifier
            synchronous: Whether to wait for completion
            
        Returns:
            Optimization execution result
        """
        try:
            # Check if optimization exists
            if optimization_id not in self.active_optimizations:
                return {
                    "success": False,
                    "error": f"Optimization {optimization_id} not found",
                }
            
            opt_data = self.active_optimizations[optimization_id]
            
            # Check status
            if opt_data["status"] == "running":
                return {
                    "success": False,
                    "error": "Optimization already running",
                    "status": "running",
                }
            
            # Set status to running
            opt_data["status"] = "running"
            opt_data["started_at"] = time.time()
            
            logger.info(f"Starting optimization {optimization_id}")
            
            if synchronous:
                return self._run_optimization_sync(optimization_id)
            else:
                self._run_optimization_async(optimization_id)
                return {
                    "success": True,
                    "optimization_id": optimization_id,
                    "status": "running",
                    "message": "Optimization started",
                }
                
        except Exception as e:
            logger.error(f"Failed to run optimization: {e}")
            if optimization_id in self.active_optimizations:
                self.active_optimizations[optimization_id]["status"] = "failed"
            
            return {
                "success": False,
                "error": str(e),
            }
    
    def _run_optimization_sync(self, optimization_id: str) -> Dict[str, Any]:
        """Run optimization synchronously."""
        try:
            self._run_optimization_async(optimization_id)
            
            # Wait for completion
            while self.active_optimizations[optimization_id]["status"] == "running":
                time.sleep(1)  # Check every second
            
            return self.get_optimization_status(optimization_id)
            
        except Exception as e:
            logger.error(f"Synchronous optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def _run_optimization_async(self, optimization_id: str):
        """Run optimization asynchronously."""
        try:
            opt_data = self.active_optimizations[optimization_id]
            opt_params = opt_data["parameters"]
            
            # Initialize candidate parameters
            opt_data["candidate_params"] = self._generate_initial_candidates(opt_params)
            
            # Run optimization loop
            while (opt_data["current_iteration"] < opt_params.max_iterations and
                   opt_data["budget_used"] < opt_params.budget and
                   opt_data["status"] == "running"):
                
                # Select candidate to evaluate
                candidate = self._select_candidate(opt_data)
                if not candidate:
                    break
                
                # Evaluate candidate
                score, evaluation_cost = self._evaluate_candidate(
                    opt_params.target_module,
                    candidate,
                    opt_params.evaluation_metric
                )
                
                # Update optimization data
                opt_data["budget_used"] += evaluation_cost
                opt_data["evaluation_history"].append({
                    "iteration": opt_data["current_iteration"],
                    "parameters": candidate,
                    "score": score,
                    "cost": evaluation_cost,
                })
                
                # Update best if improved
                is_better = (
                    (opt_params.evaluation_metric != "loss" and score > opt_data["best_score"]) or
                    (opt_params.evaluation_metric == "loss" and score < opt_data["best_score"])
                )
                
                if is_better:
                    opt_data["best_score"] = score
                    opt_data["best_params"] = candidate
                
                # Generate new candidates based on strategy
                self._update_candidates(opt_data, candidate, score)
                
                opt_data["current_iteration"] += 1
            
            # Finalize optimization
            opt_data["status"] = "completed"
            opt_data["completed_at"] = time.time()
            opt_data["improvement"] = self._calculate_improvement(opt_data)
            
            # Move to history
            self.optimization_history[optimization_id] = opt_data
            del self.active_optimizations[optimization_id]
            
            logger.info(f"Optimization {optimization_id} completed. Best score: {opt_data['best_score']}")
            
        except Exception as e:
            logger.error(f"Async optimization failed: {e}")
            if optimization_id in self.active_optimizations:
                self.active_optimizations[optimization_id]["status"] = "failed"
                self.active_optimizations[optimization_id]["error"] = str(e)
                self.active_optimizations[optimization_id]["completed_at"] = time.time()
                
                # Move to history
                self.optimization_history[optimization_id] = self.active_optimizations[optimization_id]
                del self.active_optimizations[optimization_id]
    
    def _validate_target_module(self, target_module: str) -> bool:
        """Validate that the target module is optimizable."""
        # List of optimizable modules
        optimizable_modules = [
            "core",
            "core.reasoning",
            "core.autonomous_planner",
            "memory",
            "persona",
            "missions",
            "optimizer",
        ]
        
        return target_module in optimizable_modules
    
    def _generate_initial_candidates(self, opt_params: OptimizationParameters) -> List[Dict[str, Any]]:
        """Generate initial candidate parameters."""
        candidates = []
        
        # Default parameter spaces based on module
        default_spaces = {
            "core": {
                "learning_rate": [0.001, 0.01, 0.1],
                "batch_size": [16, 32, 64],
                "dropout": [0.1, 0.2, 0.3],
                "hidden_size": [128, 256, 512],
            },
            "core.reasoning": {
                "temperature": [0.5, 0.7, 0.9],
                "max_tokens": [100, 500, 1000],
                "top_k": [40, 50, 60],
                "top_p": [0.8, 0.9, 0.95],
            },
            "memory": {
                "recall_top_k": [3, 5, 10],
                "min_similarity": [0.1, 0.2, 0.3],
                "max_context_tokens": [50, 100, 200],
            },
        }
        
        param_space = opt_params.parameter_space or default_spaces.get(opt_params.target_module, {})
        
        # Generate candidates based on strategy
        if opt_params.strategy == OptimizationStrategy.RANDOM_SEARCH:
            candidates = self._generate_random_candidates(param_space, 10)
        elif opt_params.strategy == OptimizationStrategy.GRID:
            candidates = self._generate_grid_candidates(param_space)
        else:
            candidates = self._generate_random_candidates(param_space, 5)
        
        return candidates
    
    def _generate_random_candidates(self, param_space: Dict[str, List[Any]], count: int) -> List[Dict[str, Any]]:
        """Generate random candidate parameters."""
        if not param_space:
            return [{} for _ in range(count)]
        
        candidates = []
        for _ in range(count):
            candidate = {}
            for param_name, param_values in param_space.items():
                candidate[param_name] = random.choice(param_values)
            candidates.append(candidate)
        
        return candidates
    
    def _generate_grid_candidates(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate candidate parameters using grid search."""
        if not param_space:
            return [{}]
        
        # Simple grid search (limit to first 2 parameters to avoid explosion)
        param_names = list(param_space.keys())[:2]
        param_values = [param_space[name] for name in param_names]
        
        candidates = []
        for combination in zip(*param_values):
            candidate = dict(zip(param_names, combination))
            candidates.append(candidate)
        
        # Add default values for other parameters
        for candidate in candidates:
            for param_name in param_space:
                if param_name not in candidate:
                    candidate[param_name] = param_space[param_name][0]
        
        return candidates
    
    def _select_candidate(self, opt_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select candidate for evaluation."""
        candidates = opt_data["candidate_params"]
        
        if not candidates:
            return None
        
        return candidates.pop(0)
    
    def _evaluate_candidate(
        self,
        target_module: str,
        candidate: Dict[str, Any],
        metric: str,
    ) -> Tuple[float, int]:
        """Evaluate a candidate parameter set.
        
        Returns:
            Tuple of (score, evaluation_cost)
        """
        # Simplified evaluation - in real implementation, this would
        # train/test the model with the candidate parameters
        try:
            # Simulate evaluation with some randomness
            base_score = 0.7  # Base performance
            improvement = random.uniform(-0.1, 0.2)  # Random improvement
            
            if metric == "loss":
                score = 1.0 - (base_score + improvement)
            else:
                score = base_score + improvement
            
            # Limit score to [0, 1] range
            score = max(0.0, min(1.0, score))
            
            # Simulated evaluation cost
            cost = random.randint(5, 20)
            
            return score, cost
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0, 1
    
    def _update_candidates(
        self,
        opt_data: Dict[str, Any],
        evaluated_candidate: Dict[str, Any],
        score: float,
    ):
        """Update candidate pool based on evaluation result."""
        opt_params = opt_data["parameters"]
        
        if opt_params.strategy == OptimizationStrategy.GENETIC:
            # Genetic algorithm - create mutations of best candidates
            if score > opt_data.get("best_score", 0):
                # Create mutations of good candidate
                mutations = self._create_mutations(evaluated_candidate, opt_params.parameter_space)
                opt_data["candidate_params"].extend(mutations[:3])
        elif opt_params.strategy == OptimizationStrategy.BAYESIAN:
            # Bayesian optimization - generate candidates near best
            if score > opt_data.get("best_score", 0):
                nearby_candidates = self._create_nearby_candidates(
                    evaluated_candidate,
                    opt_params.parameter_space
                )
                opt_data["candidate_params"].extend(nearby_candidates[:2])
        else:
            # Random search - generate new random candidates
            new_candidates = self._generate_random_candidates(
                opt_params.parameter_space,
                2
            )
            opt_data["candidate_params"].extend(new_candidates)
    
    def _create_mutations(
        self,
        base_candidate: Dict[str, Any],
        param_space: Dict[str, List[Any]],
        mutation_count: int = 5,
    ) -> List[Dict[str, Any]]:
        """Create mutations of a base candidate."""
        mutations = []
        
        for _ in range(mutation_count):
            mutation = base_candidate.copy()
            
            # Mutate 1-2 parameters
            num_mutations = random.randint(1, min(2, len(base_candidate)))
            params_to_mutate = random.sample(list(base_candidate.keys()), num_mutations)
            
            for param in params_to_mutate:
                if param in param_space:
                    # Choose a different value
                    current_value = base_candidate[param]
                    valid_values = [v for v in param_space[param] if v != current_value]
                    if valid_values:
                        mutation[param] = random.choice(valid_values)
            
            mutations.append(mutation)
        
        return mutations
    
    def _create_nearby_candidates(
        self,
        base_candidate: Dict[str, Any],
        param_space: Dict[str, List[Any]],
        count: int = 3,
    ) -> List[Dict[str, Any]]:
        """Create candidates with parameters near the base candidate."""
        nearby = []
        
        for _ in range(count):
            candidate = base_candidate.copy()
            
            # Adjust 1 parameter slightly
            for param_name, param_value in base_candidate.items():
                if param_name in param_space and isinstance(param_value, (int, float)):
                    values = param_space[param_name]
                    if all(isinstance(v, (int, float)) for v in values):
                        # Find index in sorted values
                        sorted_values = sorted(values)
                        try:
                            idx = sorted_values.index(param_value)
                            
                            # Choose adjacent value if available
                            if random.random() < 0.5 and idx > 0:
                                candidate[param_name] = sorted_values[idx - 1]
                            elif idx < len(sorted_values) - 1:
                                candidate[param_name] = sorted_values[idx + 1]
                        except ValueError:
                            pass
            
            nearby.append(candidate)
        
        return nearby
    
    def _calculate_improvement(self, opt_data: Dict[str, Any]) -> float:
        """Calculate improvement over baseline."""
        if not opt_data["evaluation_history"]:
            return 0.0
        
        baseline_score = opt_data["evaluation_history"][0]["score"]
        best_score = opt_data["best_score"]
        
        if opt_data["parameters"].evaluation_metric == "loss":
            # Lower is better for loss
            return (baseline_score - best_score) / max(abs(baseline_score), 1e-8)
        else:
            # Higher is better for other metrics
            return (best_score - baseline_score) / max(abs(baseline_score), 1e-8)
    
    def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """Get current status of an optimization."""
        try:
            # Check active optimizations
            if optimization_id in self.active_optimizations:
                opt_data = self.active_optimizations[optimization_id]
            # Check optimization history
            elif optimization_id in self.optimization_history:
                opt_data = self.optimization_history[optimization_id]
            else:
                return {
                    "success": False,
                    "error": f"Optimization {optimization_id} not found",
                }
            
            # Build status response
            response = {
                "success": True,
                "optimization_id": optimization_id,
                "status": opt_data["status"],
                "target_module": opt_data["parameters"].target_module,
                "optimization_type": opt_data["parameters"].optimization_type.value,
                "strategy": opt_data["parameters"].strategy.value,
                "current_iteration": opt_data["current_iteration"],
                "max_iterations": opt_data["parameters"].max_iterations,
                "budget_used": opt_data["budget_used"],
                "budget": opt_data["parameters"].budget,
                "best_score": opt_data["best_score"],
                "best_params": opt_data["best_params"],
            }
            
            if opt_data.get("created_at"):
                response["created_at"] = opt_data["created_at"]
            
            if opt_data.get("started_at"):
                response["started_at"] = opt_data["started_at"]
            
            if opt_data.get("completed_at"):
                response["completed_at"] = opt_data["completed_at"]
                response["improvement"] = opt_data.get("improvement", 0.0)
                response["execution_time"] = response["completed_at"] - response["started_at"]
            
            if opt_data.get("error"):
                response["error"] = opt_data["error"]
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get optimization status: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def list_optimizations(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """List optimizations with optional status filter."""
        try:
            all_optimizations = {
                **self.active_optimizations,
                **self.optimization_history
            }
            
            # Apply status filter
            if status_filter:
                all_optimizations = {
                    opt_id: opt_data
                    for opt_id, opt_data in all_optimizations.items()
                    if opt_data["status"] == status_filter
                }
            
            # Build response
            optimizations_list = []
            for opt_id, opt_data in all_optimizations.items():
                opt_info = {
                    "optimization_id": opt_id,
                    "target_module": opt_data["parameters"].target_module,
                    "type": opt_data["parameters"].optimization_type.value,
                    "status": opt_data["status"],
                    "best_score": opt_data["best_score"],
                }
                
                if opt_data.get("started_at"):
                    opt_info["started_at"] = opt_data["started_at"]
                
                if opt_data.get("completed_at"):
                    opt_info["completed_at"] = opt_data["completed_at"]
                    opt_info["improvement"] = opt_data.get("improvement", 0.0)
                
                optimizations_list.append(opt_info)
            
            return {
                "success": True,
                "optimizations": optimizations_list,
                "total_returned": len(optimizations_list),
                "total_optimizations": len(self.active_optimizations) + len(self.optimization_history),
            }
            
        except Exception as e:
            logger.error(f"Failed to list optimizations: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Modal function wrapper for optimization
def ada_optimize(params: Dict[str, Any]) -> Dict[str, Any]:
    """Modal optimization function wrapper."""
    try:
        service = AdaOptimizerService()
        
        # Create and run optimization in one call
        creation_result = service.create_optimization(params)
        
        if not creation_result["success"]:
            return creation_result
        
        optimization_id = creation_result["optimization_id"]
        
        # Execute synchronously
        execution_result = service.run_optimization(optimization_id, synchronous=True)
        
        return execution_result
        
    except Exception as e:
        logger.error(f"Modal optimization failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    # Local testing
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Ada Optimizer Service")
    parser.add_argument("--command", default="run", choices=["run", "status", "list"])
    parser.add_argument("--target-module", default="core", help="Target module to optimize")
    parser.add_argument("--type", default="parameter_tuning", help="Optimization type")
    parser.add_argument("--budget", type=int, default=1000, help="Optimization budget")
    parser.add_argument("--optimization-id", help="Optimization ID for status action")
    
    args = parser.parse_args()
    
    service = AdaOptimizerService()
    
    if args.command == "run":
        params = {
            "target_module": args.target_module,
            "type": args.type,
            "budget": args.budget,
        }
        result = ada_optimize(params)
    elif args.command == "status":
        if not args.optimization_id:
            result = {"error": "Optimization ID required for status check"}
        else:
            result = service.get_optimization_status(args.optimization_id)
    elif args.command == "list":
        result = service.list_optimizations()
    
    print(json.dumps(result, indent=2))
