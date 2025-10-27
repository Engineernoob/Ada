"""Executor for running plans and managing tool execution with reward feedback."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pathlib import Path

from tools.registry import tool_registry

if TYPE_CHECKING:  # pragma: no cover - imported only for typing
    from planner.planner import Plan, PlanStep


@dataclass
class ExecutionResult:
    """Result of executing a plan step or entire plan."""
    step_id: Optional[str]
    plan_id: str
    success: bool
    data: Any
    error: Optional[str]
    timestamp: float
    execution_time: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "plan_id": self.plan_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
            "execution_time": self.execution_time
        }


class Executor:
    """Executes plans and manages tool execution with reward feedback."""

    def __init__(self, confirm_before_write: bool = True) -> None:
        self.confirm_before_write = confirm_before_write
        self.execution_history: List[ExecutionResult] = []
        self.current_plans: Dict[str, Dict] = {}  # Track running plans
        self._storage_path = Path(__file__).resolve().parents[1] / "storage" / "plans.json"
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._plan_store: Dict[str, Dict[str, Any]] = {}
        self._plans: Dict[str, "Plan"] = {}
        self._plan_summaries: Dict[str, Dict[str, Any]] = {}
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Plan registration and lookup helpers
    # ------------------------------------------------------------------
    def register_plan(self, plan: "Plan") -> None:
        """Register a plan so it can be executed or inspected later."""

        self._plans[plan.id] = plan
        serialized = self._serialize_plan(plan)
        self._plan_store[plan.id] = serialized
        self._plan_summaries[plan.id] = self._summary_from_serialized(serialized)
        self._persist()

    def get_plan(self, plan_id: str) -> Optional["Plan"]:
        if plan_id in self._plans:
            return self._plans[plan_id]

        data = self._plan_store.get(plan_id)
        if not data:
            return None

        plan = self._deserialize_plan(data)
        self._plans[plan_id] = plan
        return plan

    def list_plans(self) -> List["Plan"]:
        return list(self._plans.values())

    def get_plan_summary(self, plan_id: str) -> Optional[Dict[str, Any]]:
        return self._plan_summaries.get(plan_id)

    def get_plan_summaries(self, limit: int | None = None) -> List[Dict[str, Any]]:
        summaries = list(self._plan_summaries.values())
        summaries.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        if limit is not None:
            return summaries[:limit]
        return summaries

    def get_plan_actions(self, plan_id: str) -> List[SimpleNamespace]:
        plan = self.get_plan(plan_id)
        if not plan:
            return []

        actions: List[SimpleNamespace] = []
        for step in plan.steps:
            actions.append(
                SimpleNamespace(
                    tool=step.tool,
                    description=getattr(step, "description", ""),
                    input_data=step.parameters,
                    duration_seconds=getattr(step, "estimated_duration", 0.0),
                )
            )
        return actions

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def run_plan(self, plan_or_id: "Plan | str") -> Dict[str, Any]:
        """Execute a plan by object or identifier and return a summary."""

        plan: Optional["Plan"]
        if isinstance(plan_or_id, str):
            plan = self.get_plan(plan_or_id)
            if plan is None:
                raise ValueError(f"Unknown plan id '{plan_or_id}'")
        else:
            plan = plan_or_id
            if plan.id not in self._plans:
                self.register_plan(plan)

        results = self.run(plan)
        summary = self._build_summary(plan, results)
        self._plan_summaries[plan.id] = summary
        self._plan_store[plan.id] = self._serialize_plan(plan)
        self._plan_store[plan.id]["results"] = summary["results"]
        self._plan_store[plan.id]["duration_seconds"] = summary["duration_seconds"]
        self._plan_store[plan.id]["output"] = summary["output"]
        self._persist()
        return summary

    def _build_summary(
        self, plan: "Plan", results: List[ExecutionResult]
    ) -> Dict[str, Any]:
        step_lookup: Dict[str, ExecutionResult] = {
            result.step_id: result for result in results if result.step_id
        }

        step_summaries: List[Dict[str, Any]] = []
        for index, step in enumerate(plan.steps, start=1):
            result = step_lookup.get(step.step_id)
            step_summary = {
                "step_number": index,
                "step_id": step.step_id,
                "tool": step.tool,
                "description": getattr(step, "description", ""),
                "success": bool(result.success if result else step.completed),
                "error": result.error if result else step.error,
                "data": result.data if result else step.result,
                "duration_seconds": result.execution_time if result else 0.0,
            }
            step_summaries.append(step_summary)

        completed_steps = sum(1 for summary in step_summaries if summary["success"])
        total_duration = sum(summary["duration_seconds"] for summary in step_summaries)
        final_output = ""
        if step_summaries:
            final_output = str(step_summaries[-1]["data"] or "")

        return {
            "id": plan.id,
            "plan_id": plan.id,
            "goal": plan.goal,
            "category": plan.category,
            "priority": plan.priority,
            "confidence": getattr(plan, "confidence", 0.0),
            "status": plan.status,
            "created_at": plan.created_at,
            "started_at": plan.started_at,
            "completed_at": plan.completed_at,
            "completed_steps": completed_steps,
            "total_steps": len(plan.steps),
            "results": step_summaries,
            "success": plan.status == "completed",
            "duration_seconds": total_duration,
            "output": final_output,
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _serialize_plan(self, plan: "Plan") -> Dict[str, Any]:
        return {
            "id": plan.id,
            "goal": plan.goal,
            "category": plan.category,
            "priority": plan.priority,
            "confidence": getattr(plan, "confidence", 0.0),
            "status": plan.status,
            "created_at": plan.created_at,
            "started_at": plan.started_at,
            "completed_at": plan.completed_at,
            "success_rate": getattr(plan, "success_rate", 0.0),
            "steps": [
                {
                    "step_id": step.step_id,
                    "tool": step.tool,
                    "action": step.action,
                    "parameters": step.parameters,
                    "description": getattr(step, "description", ""),
                    "dependencies": step.dependencies,
                    "completed": step.completed,
                    "result": step.result,
                    "error": step.error,
                    "timestamp": step.timestamp,
                    "estimated_duration": getattr(step, "estimated_duration", 0.0),
                }
                for step in plan.steps
            ],
        }

    def _deserialize_plan(self, data: Dict[str, Any]) -> "Plan":
        from planner.planner import Plan, PlanStep

        steps: List[PlanStep] = []
        for step_data in data.get("steps", []):
            step = PlanStep(
                step_id=step_data.get("step_id", str(uuid.uuid4())),
                tool=step_data.get("tool", "unknown"),
                action=step_data.get("action", ""),
                parameters=step_data.get("parameters", {}),
                description=step_data.get("description", ""),
                dependencies=step_data.get("dependencies", []),
                completed=step_data.get("completed", False),
                result=step_data.get("result"),
                error=step_data.get("error"),
                timestamp=step_data.get("timestamp", time.time()),
                estimated_duration=step_data.get("estimated_duration", 0.0),
            )
            steps.append(step)

        plan = Plan(
            id=data.get("id", str(uuid.uuid4())),
            goal=data.get("goal", ""),
            category=data.get("category", "action"),
            priority=float(data.get("priority", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            steps=steps,
            status=data.get("status", "created"),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            success_rate=float(data.get("success_rate", 0.0)),
        )
        return plan

    def _summary_from_serialized(self, data: Dict[str, Any]) -> Dict[str, Any]:
        steps = data.get("steps", [])
        completed_steps = sum(1 for step in steps if step.get("completed"))
        return {
            "id": data.get("id", ""),
            "goal": data.get("goal", ""),
            "category": data.get("category", ""),
            "priority": float(data.get("priority", 0.0)),
            "confidence": float(data.get("confidence", 0.0)),
            "status": data.get("status", "created"),
            "created_at": data.get("created_at", 0.0),
            "started_at": data.get("started_at"),
            "completed_at": data.get("completed_at"),
            "completed_steps": completed_steps,
            "total_steps": len(steps),
            "results": data.get("results", []),
            "success": data.get("status") == "completed",
            "duration_seconds": float(data.get("duration_seconds", 0.0)),
            "output": data.get("output", ""),
        }

    def _persist(self) -> None:
        payload = {"plans": list(self._plan_store.values())}
        try:
            self._storage_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def _load_from_disk(self) -> None:
        if not self._storage_path.exists():
            return

        try:
            content = self._storage_path.read_text(encoding="utf-8")
            data = json.loads(content) if content.strip() else {}
        except (OSError, json.JSONDecodeError):
            return

        plans = data.get("plans", []) if isinstance(data, dict) else []
        for plan_data in plans:
            if not isinstance(plan_data, dict):
                continue
            plan_id = plan_data.get("id")
            if not plan_id:
                continue
            self._plan_store[plan_id] = plan_data
            self._plan_summaries[plan_id] = self._summary_from_serialized(plan_data)
        
    def run(self, plan) -> List[ExecutionResult]:
        """
        Execute a complete plan.
        
        Args:
            plan: Plan object to execute
            
        Returns:
            List of execution results for each step
        """
        results = []
        plan.status = "executing"
        plan.started_at = time.time()
        
        # Track this plan
        self.current_plans[plan.id] = {
            "plan": plan,
            "start_time": plan.started_at,
            "total_steps": len(plan.steps),
            "completed_steps": 0
        }
        
        try:
            while True:
                next_step = plan.get_next_step()
                if not next_step:
                    break  # Plan complete
            
                step_result = self.execute_step(plan, next_step)
                results.append(step_result)
                
                # Update step status in plan
                next_step.completed = step_result.success
                next_step.result = str(step_result.data) if step_result.success else None
                next_step.error = step_result.error if not step_result.success else None
                
                # Update tracking
                if step_result.success:
                    self.current_plans[plan.id]["completed_steps"] += 1
                
                # Store execution result
                self.execution_history.append(step_result)
                
                # Stop execution if critical step fails
                if not step_result.success and self._is_critical_step(next_step):
                    plan.status = "failed"
                    break
        
        except Exception as e:
            plan.status = "failed"
            error_result = ExecutionResult(
                step_id=None,
                plan_id=plan.id,
                success=False,
                data=None,
                error=f"Plan execution failed: {str(e)}",
                timestamp=time.time(),
                execution_time=0.0
            )
            results.append(error_result)
        
        # Update plan final status
        if plan.status == "executing":
            plan.status = "completed"
        
        plan.completed_at = time.time()
        plan.update_success_rate()
        
        # Clean up tracking
        if plan.id in self.current_plans:
            del self.current_plans[plan.id]
        
        return results
    
    def execute_step(self, plan, step) -> ExecutionResult:
        """
        Execute a single plan step.
        
        Args:
            plan: Plan object
            step: PlanStep to execute
            
        Returns:
            Execution result for the step
        """
        start_time = time.time()
        
        try:
            # Prepare parameters for tool execution
            tool_params = self._prepare_parameters(plan, step)
            
            # Check for confirmation requirements
            if self._requires_confirmation(step):
                confirmed = self._request_confirmation(step, tool_params)
                if not confirmed:
                    return ExecutionResult(
                        step_id=step.step_id,
                        plan_id=plan.id,
                        success=False,
                        data=None,
                        error="Execution cancelled by user confirmation requirement",
                        timestamp=start_time,
                        execution_time=0.0
                    )
            
            # Execute the tool
            result = tool_registry.execute(step.tool, step.action, **tool_params)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                step_id=step.step_id,
                plan_id=plan.id,
                success=result.success,
                data=result.data,
                error=result.error,
                timestamp=start_time,
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                step_id=step.step_id,
                plan_id=plan.id,
                success=False,
                data=None,
                error=f"Step execution failed: {str(e)}",
                timestamp=start_time,
                execution_time=execution_time
            )
    
    def _prepare_parameters(self, plan, step) -> Dict[str, Any]:
        """Prepare parameters for tool execution, resolving step dependencies."""
        params = step.parameters.copy()
        
        # Resolve parameter references (e.g., "step_id.result")
        for key, value in params.items():
            if isinstance(value, str) and "." in value:
                parts = value.split(".")
                if len(parts) == 2 and parts[1] in ["result", "data", "output"]:
                    dep_step_id = parts[0]
                    dep_step = plan._get_step_by_id(dep_step_id)
                    if dep_step and dep_step.completed and dep_step.result:
                        params[key] = dep_step.result
        
        # Add context information
        params.update({
            "plan_id": plan.id,
            "goal": plan.goal,
            "description": step.parameters.get("description", "")
        })
        
        return params
    
    def _requires_confirmation(self, step) -> bool:
        """Check if a step requires user confirmation."""
        if not self.confirm_before_write:
            return False
        
        # Write operations typically require confirmation
        if step.tool == "file_ops" and step.action in ["write", "move", "delete", "copy"]:
            return True
        
        # Large data operations might require confirmation
        if step.action in ["summarize", "analyze"] and self._is_large_operation(step):
            return True
        
        return False
    
    def _is_large_operation(self, step) -> bool:
        """Check if this is a potentially large operation."""
        # Simple heuristic - if it might process lots of data
        return step.action in ["summarize", "analyze", "extract"]
    
    def _request_confirmation(self, step, params: Dict[str, Any]) -> bool:
        """Request user confirmation for a step."""
        # In a real implementation, this would prompt the user
        # For now, auto-confirm safe operations
        auto_confirm_tools = {
            "read", "list", "search", "evaluate", "interpret", "extract", "run", "write"
        }
        
        if step.action in auto_confirm_tools:
            return True
        
        # Default to confirm for potentially dangerous operations
        return False
    
    def _is_critical_step(self, step) -> bool:
        """Check if a step is critical to plan success."""
        critical_actions = {"write", "copy", "move", "create", "build"}
        return step.action in critical_actions
    
    def calculate_reward(self, results: List[ExecutionResult]) -> float:
        """
        Calculate reward for plan execution based on success rates and other factors.
        
        Args:
            results: List of execution results
            
        Returns:
            Calculated reward value
        """
        if not results:
            return 0.0
        
        # Base reward from success rate
        successful_steps = sum(1 for r in results if r.success)
        total_steps = len(results)
        success_rate = successful_steps / total_steps
        
        # Adjust for execution time (faster is better)
        total_time = sum(r.execution_time for r in results)
        time_factor = max(0.1, 1.0 - (total_time / 60.0))  # Penalize if > 60 seconds
        
        # Adjust for step complexity
        complexity_bonus = min(0.2, total_steps * 0.05)
        
        # Calculate final reward
        reward = success_rate * 0.7 + time_factor * 0.2 + complexity_bonus * 0.1
        
        # Bonus for perfect execution
        if success_rate == 1.0:
            reward += 0.1
        
        return min(1.0, reward)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about execution history."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_time": 0.0,
                "current_plans": 0
            }
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for r in self.execution_history if r.success)
        success_rate = successful_executions / total_executions
        average_time = sum(r.execution_time for r in self.execution_history) / total_executions
        current_plans = len(self.current_plans)
        
        # Steps by tool
        steps_by_tool = {}
        for result in self.execution_history:
            tool_name = result.step_id.split("_")[0] if result.step_id else "unknown"
            if tool_name not in steps_by_tool:
                steps_by_tool[tool_name] = {"total": 0, "successful": 0}
            steps_by_tool[tool_name]["total"] += 1
            if result.success:
                steps_by_tool[tool_name]["successful"] += 1
        
        return {
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_time": average_time,
            "current_plans": current_plans,
            "steps_by_tool": steps_by_tool
        }
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a currently running plan."""
        if plan_id not in self.current_plans:
            return None
        
        plan_info = self.current_plans[plan_id]
        plan = plan_info["plan"]
        
        return {
            "plan_id": plan_id,
            "goal": plan.goal,
            "status": plan.status,
            "total_steps": plan_info["total_steps"],
            "completed_steps": plan_info["completed_steps"],
            "progress": plan_info["completed_steps"] / plan_info["total_steps"],
            "start_time": plan_info["start_time"],
            "elapsed_time": time.time() - plan_info["start_time"]
        }
    
    def abort_plan(self, plan_id: str) -> bool:
        """Abort a currently running plan."""
        if plan_id not in self.current_plans:
            return False
        
        plan_info = self.current_plans[plan_id]
        plan = plan_info["plan"]
        
        plan.status = "aborted"
        plan.completed_at = time.time()
        
        # Clean up tracking
        del self.current_plans[plan_id]
        
        return True
    
    def set_confirmation_mode(self, confirm_before_write: bool) -> None:
        """Set the confirmation mode for write operations."""
        self.confirm_before_write = confirm_before_write
