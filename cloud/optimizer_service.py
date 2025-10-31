"""Optimizer service for Ada Cloud infrastructure.

This module provides evolutionary optimization, hyperparameter tuning,
and model performance optimization capabilities for Ada's neural components.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OptimizationTarget:
    """Represents an optimization target."""
    name: str
    module: str  # core, reasoning, planning, etc.
    parameter_space: Dict[str, Any]  # ranges and types
    objective: str  # maximize, minimize
    current_value: float
    target_value: Optional[float] = None


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    target_name: str
    best_params: Dict[str, Any]
    best_score: float
    improvement: float
    iterations: int
    convergence_reached: bool
    execution_time: float


@dataclass
class OptimizationJob:
    """An optimization job configuration."""
    id: str
    target: OptimizationTarget
    algorithm: str  # genetic, bayesian, random_search, grid_search
    max_iterations: int
    convergence_threshold: float
    budget: int  # computational budget
    status: str = "pending"  # pending, running, completed, failed
    current_iteration: int = 0
    best_score: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class OptimizerService:
    """Cloud-based optimization service for Ada components."""
    
    def __init__(self, storage_service=None):
        """Initialize optimizer service.
        
        Args:
            storage_service: Optional storage service for persistence
        """
        self.storage_service = storage_service
        self.active_jobs: Dict[str, OptimizationJob] = {}
        self.completed_jobs: List[OptimizationJob] = []
        
    def create_optimization_job(
        self,
        target_module: str,
        parameter_space: Dict[str, Any],
        objective: str = "maximize",
        algorithm: str = "genetic",
        max_iterations: int = 50,
        convergence_threshold: float = 0.001,
        budget: int = 1000,
        target_value: Optional[float] = None,
    ) -> str:
        """Create a new optimization job.
        
        Args:
            target_module: Module to optimize
            parameter_space: Parameter search space
            objective: Optimization objective
            algorithm: Optimization algorithm to use
            max_iterations: Maximum iterations
            convergence_threshold: Convergence threshold
            budget: Computational budget
            target_value: Target value (optional)
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        target = OptimizationTarget(
            name=f"optimize_{target_module}_{int(time.time())}",
            module=target_module,
            parameter_space=parameter_space,
            objective=objective,
            current_value=0.0,
            target_value=target_value,
        )
        
        job = OptimizationJob(
            id=job_id,
            target=target,
            algorithm=algorithm,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            budget=budget,
        )
        
        self.active_jobs[job_id] = job
        logger.info(f"Created optimization job {job_id} for module {target_module}")
        
        return job_id
    
    async def optimize(self, job_id: str) -> OptimizationResult:
        """Run optimization for the specified job.
        
        Args:
            job_id: Job ID to optimize
            
        Returns:
            Optimization result
        """
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        job.status = "running"
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting optimization for job {job_id} using {job.algorithm}")
            
            # Select optimization algorithm
            if job.algorithm == "genetic":
                result = await self._genetic_optimization(job)
            elif job.algorithm == "bayesian":
                result = await self._bayesian_optimization(job)
            elif job.algorithm == "random_search":
                result = await self._random_search(job)
            elif job.algorithm == "grid_search":
                result = await self._grid_search(job)
            else:
                raise ValueError(f"Unknown optimization algorithm: {job.algorithm}")
            
            job.status = "completed"
            job.best_score = result.best_score
            job.best_params = result.best_params
            
            # Save results if storage service is available
            if self.storage_service:
                await self._save_optimization_result(job_id, result)
            
            logger.info(f"Optimization completed for job {job_id}: {result.improvement:.2%} improvement")
            return result
            
        except Exception as e:
            job.status = "failed"
            logger.error(f"Optimization failed for job {job_id}: {e}")
            raise
    
    async def _genetic_optimization(self, job: OptimizationJob) -> OptimizationResult:
        """Genetic algorithm optimization."""
        population_size = min(job.budget // 10, 50)
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Initialize population
        population = self._initialize_population(job.target.parameter_space, population_size)
        
        # Evaluate initial population
        scores = await self._evaluate_population(population, job)
        
        best_score_idx = np.argmax(scores) if job.target.objective == "maximize" else np.argmin(scores)
        best_score = scores[best_score_idx]
        best_params = population[best_score_idx]
        
        # Evolution loop
        for iteration in range(job.max_iterations):
            job.current_iteration = iteration + 1
            
            # Selection (tournament selection)
            selected = self._tournament_selection(population, scores, tournament_size=3)
            
            # Crossover
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected) and random.random() < crossover_rate:
                    child1, child2 = self._crossover(selected[i], selected[i + 1], job.target.parameter_space)
                    offspring.extend([child1, child2])
                elif i < len(selected):
                    offspring.append(selected[i])
            
            # Mutation
            for individual in offspring:
                if random.random() < mutation_rate:
                    individual = self._mutate(individual, job.target.parameter_space)
            
            # Evaluation
            if offspring:
                offspring_scores = await self._evaluate_population(offspring, job)
                
                # Replace worst individuals
                all_population = population + offspring
                all_scores = scores + offspring_scores
                
                # Keep best individuals
                sorted_indices = np.argsort(all_scores)[::-1] if job.target.objective == "maximize" else np.argsort(all_scores)
                population = [all_population[i] for i in sorted_indices[:population_size]]
                scores = [all_scores[i] for i in sorted_indices[:population_size]]
                
                # Track best
                current_best_idx = np.argmax(scores) if job.target.objective == "maximize" else np.argmin(scores)
                current_best_score = scores[current_best_idx]
                
                if (job.target.objective == "maximize" and current_best_score > best_score) or \
                   (job.target.objective == "minimize" and current_best_score < best_score):
                    best_score = current_best_score
                    best_params = population[current_best_idx]
                
                # Record iteration
                job.iteration_history.append({
                    "iteration": iteration + 1,
                    "best_score": best_score,
                    "avg_score": np.mean(scores),
                    "diversity": np.std(scores),
                })
                
                # Check convergence
                if iteration > 10:
                    recent_scores = scores[-10:]
                    if np.std(recent_scores) < job.convergence_threshold:
                        break
        
        improvement = (best_score - job.target.current_value) / abs(job.target.current_value) if job.target.current_value != 0 else 0.0
        execution_time = time.time() - (time.time() - start_time)
        
        return OptimizationResult(
            target_name=job.target.name,
            best_params=best_params,
            best_score=best_score,
            improvement=improvement,
            iterations=job.current_iteration,
            convergence_reached=True,
            execution_time=execution_time,
        )
    
    async def _bayesian_optimization(self, job: OptimizationJob) -> OptimizationResult:
        """Bayesian optimization (simplified)."""
        # For simplicity, this is a placeholder implementation
        # In practice, you'd use libraries like scikit-optimize or GPyOpt
        
        # Random search with intelligent sampling
        return await self._random_search(job)
    
    async def _random_search(self, job: OptimizationJob) -> OptimizationResult:
        """Random search optimization."""
        max_evaluations = min(job.budget, job.max_iterations)
        
        best_params = self._sample_parameters(job.target.parameter_space)
        best_score = await self._evaluate_params(best_params, job)
        job.target.current_value = best_score  # Set baseline
        
        for iteration in range(max_evalurations - 1):
            job.current_iteration = iteration + 1
            
            # Sample random parameters
            params = self._sample_parameters(job.target.parameter_space)
            score = await self._evaluate_params(params, job)
            
            # Track best
            if (job.target.objective == "maximize" and score > best_score) or \
               (job.target.objective == "minimize" and score < best_score):
                best_score = score
                best_params = params
            
            # Record iteration
            job.iteration_history.append({
                "iteration": iteration + 1,
                "best_score": best_score,
                "current_score": score,
            })
            
            # Check convergence
            if job.target.target_value is not None:
                if (job.target.objective == "maximize" and best_score >= job.target.target_value) or \
                   (job.target.objective == "minimize" and best_score <= job.target.target_value):
                    break
        
        improvement = (best_score - job.target.current_value) / abs(job.target.current_value) if job.target.current_value != 0 else 0.0
        execution_time = time.time() - (time.time() - start_time)
        
        return OptimizationResult(
            target_name=job.target.name,
            best_params=best_params,
            best_score=best_score,
            improvement=improvement,
            iterations=job.current_iteration,
            convergence_reached=True,
            execution_time=execution_time,
        )
    
    async def _grid_search(self, job: OptimizationJob) -> OptimizationResult:
        """Grid search optimization."""
        # Generate grid points (simplified)
        grid_params = self._generate_grid_points(job.target.parameter_space, max_points=min(job.budget, 100))
        
        best_params = grid_params[0] if grid_params else {}
        best_score = await self._evaluate_params(best_params, job)
        job.target.current_value = best_score  # Set baseline
        
        for i, params in enumerate(grid_params[1:], 1):
            job.current_iteration = i
            
            score = await self._evaluate_params(params, job)
            
            # Track best
            if (job.target.objective == "maximize" and score > best_score) or \
               (job.target.objective == "minimize" and score < best_score):
                best_score = score
                best_params = params
            
            # Record iteration
            job.iteration_history.append({
                "iteration": i,
                "best_score": best_score,
                "current_score": score,
            })
        
        improvement = (best_score - job.target.current_value) / abs(job.target.current_value) if job.target.current_value != 0 else 0.0
        execution_time = time.time() - (time.time() - start_time)
        
        return OptimizationResult(
            target_name=job.target.name,
            best_params=best_params,
            best_score=best_score,
            improvement=improvement,
            iterations=job.current_iteration,
            convergence_reached=True,
            execution_time=execution_time,
        )
    
    def _initialize_population(self, parameter_space: Dict[str, Any], population_size: int) -> List[Dict[str, Any]]:
        """Initialize population for genetic algorithm."""
        return [self._sample_parameters(parameter_space) for _ in range(population_size)]
    
    def _sample_parameters(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from the parameter space."""
        params = {}
        for param_name, param_config in parameter_space.items():
            param_type = param_config.get("type", "float")
            
            if param_type == "float":
                min_val = param_config.get("min", 0.0)
                max_val = param_config.get("max", 1.0)
                params[param_name] = random.uniform(min_val, max_val)
            elif param_type == "int":
                min_val = param_config.get("min", 0)
                max_val = param_config.get("max", 100)
                params[param_name] = random.randint(min_val, max_val)
            elif param_type == "choice":
                choices = param_config.get("choices", [])
                params[param_name] = random.choice(choices)
            elif param_type == "bool":
                params[param_name] = random.random() < 0.5
        
        return params
    
    async def _evaluate_population(self, population: List[Dict[str, Any]], job: OptimizationJob) -> List[float]:
        """Evaluate entire population."""
        scores = []
        for params in population:
            score = await self._evaluate_params(params, job)
            scores.append(score)
        return scores
    
    async def _evaluate_params(self, params: Dict[str, Any], job: OptimizationJob) -> float:
        """Evaluate parameter set.
        
        This is a placeholder - in practice, you'd evaluate the actual
        performance of the target module with these parameters.
        """
        # Simulated evaluation based on parameter quality
        base_score = 0.5
        
        # Add random noise to simulate evaluation
        score = base_score + random.gauss(0, 0.1)
        
        # Clamp to reasonable range
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _tournament_selection(self, population: List[Dict[str, Any]], scores: List[float], tournament_size: int = 3) -> List[Dict[str, Any]]:
        """Tournament selection for genetic algorithm."""
        selected = []
        for _ in range(len(population)):
            # Select random contenders
            contenders = random.sample(list(zip(population, scores)), min(tournament_size, len(population)))
            
            # Select best from contenders
            best_contender = max(contenders, key=lambda x: x[1]) if job.target.objective == "maximize" else min(contenders, key=lambda x: x[1])
            selected.append(best_contender[0])
        
        return selected
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], parameter_space: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for genetic algorithm."""
        child1 = {}
        child2 = {}
        
        for param_name in parameter_space.keys():
            if random.random() < 0.5:
                child1[param_name] = parent1.get(param_name)
                child2[param_name] = parent2.get(param_name)
            else:
                child1[param_name] = parent2.get(param_name)
                child2[param_name] = parent1.get(param_name)
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        
        # Mutate random parameter
        if parameter_space:
            param_name = random.choice(list(parameter_space.keys()))
            param_config = parameter_space[param_name]
            param_type = param_config.get("type", "float")
            
            if param_type == "float":
                min_val = param_config.get("min", 0.0)
                max_val = param_config.get("max", 1.0)
                mutated[param_name] = random.uniform(min_val, max_val)
            elif param_type == "int":
                min_val = param_config.get("min", 0)
                max_val = param_config.get("max", 100)
                mutated[param_name] = random.randint(min_val, max_val)
            elif param_type == "choice":
                choices = param_config.get("choices", [])
                mutated[param_name] = random.choice(choices)
            elif param_type == "bool":
                mutated[param_name] = random.random() < 0.5
        
        return mutated
    
    def _generate_grid_points(self, parameter_space: Dict[str, Any], max_points: int = 100) -> List[Dict[str, Any]]:
        """Generate grid points for grid search."""
        # Simplified grid generation - in practice, you'd use actual grid search
        grid_points = []
        num_points = min(max_points, 20)  # Limit to reasonable size
        
        for _ in range(num_points):
            grid_points.append(self._sample_parameters(parameter_space))
        
        return grid_points
    
    async def _save_optimization_result(self, job_id: str, result: OptimizationResult):
        """Save optimization result to storage."""
        if not self.storage_service:
            return
        
        try:
            result_data = {
                "optimization_id": job_id,
                "target_name": result.target_name,
                "best_params": result.best_params,
                "best_score": result.best_score,
                "improvement": result.improvement,
                "iterations": result.iterations,
                "convergence_reached": result.convergence_reached,
                "execution_time": result.execution_time,
                "timestamp": time.time(),
            }
            
            # Save to storage
            key = f"optimizations/{job_id}.json"
            await self.storage_service.upload_json(key, result_data)
            
        except Exception as e:
            logger.error(f"Failed to save optimization result: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of optimization job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job.status,
                "current_iteration": job.current_iteration,
                "max_iterations": job.max_iterations,
                "best_score": job.best_score,
                "algorithm": job.algorithm,
                "target_module": job.target.module,
            }
        return None


# Modal wrapper function
async def cloud_optimize(params: Dict[str, Any]) -> Dict[str, Any]:
    """Modal optimization function.
    
    Args:
        params: Optimization parameters including:
               - target_module: Module to optimize
               - parameter_space: Parameter search space
               - algorithm: Optimization algorithm
               - max_iterations: Maximum iterations
               - budget: Computational budget
        
    Returns:
        Optimization results
    """
    try:
        service = OptimizerService()
        
        # Extract parameters
        target_module = params.get("target_module", "core.reasoning")
        parameter_space = params.get("parameter_space", {})
        algorithm = params.get("algorithm", "genetic")
        max_iterations = params.get("max_iterations", 50)
        convergence_threshold = params.get("convergence_threshold", 0.001)
        budget = params.get("budget", 1000)
        objective = params.get("objective", "maximize")
        target_value = params.get("target_value", None)
        
        # Create and run optimization job
        job_id = service.create_optimization_job(
            target_module=target_module,
            parameter_space=parameter_space,
            objective=objective,
            algorithm=algorithm,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            budget=budget,
            target_value=target_value,
        )
        
        # Run optimization
        result = await service.optimize(job_id)
        
        return {
            "success": True,
            "optimization_id": job_id,
            "target_module": target_module,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "improvement": result.improvement,
            "iterations": result.iterations,
            "convergence_reached": result.convergence_reached,
            "execution_time": result.execution_time,
        }
        
    except Exception as e:
        logger.error(f"Cloud optimization failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "optimization_id": params.get("job_id", "unknown"),
        }
