"""Mission execution service for Ada Cloud infrastructure.

This module provides mission orchestration and execution capabilities
optimized for Modal's serverless environment.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MissionStatus(Enum):
    """Mission execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MissionStepStatus(Enum):
    """Individual mission step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AdaMissionService:
    """Mission execution service for Ada Cloud."""
    
    def __init__(self, storage_base_path: str = "/root/ada/storage"):
        """Initialize mission service.
        
        Args:
            storage_base_path: Base path for persistent storage
        """
        self.storage_base_path = storage_base_path
        self.active_missions = {}
        self.mission_history = {}
        
        # Initialize mission manager
        self._initialize_mission_system()
    
    def _initialize_mission_system(self):
        """Initialize the mission management system."""
        try:
            from missions.mission_manager import MissionManager
            from missions.mission_manager import Mission
            
            self.mission_manager = MissionManager()
            self.Mission = Mission
            logger.info("Mission management system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize mission system: {e}")
            raise
    
    def create_mission(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "medium",
        deadline: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Create a new mission.
        
        Args:
            goal: Mission goal description
            context: Mission context and constraints
            priority: Mission priority (low/medium/high/critical)
            deadline: Optional deadline for completion
            
        Returns:
            Mission creation result
        """
        try:
            mission_id = str(uuid.uuid4())
            
            # Validate inputs
            if not goal or len(goal.strip()) == 0:
                return {
                    "success": False,
                    "error": "Mission goal cannot be empty",
                }
            
            # Create mission object
            mission = self.Mission(
                goal=goal,
                context=context or {},
                priority=priority,
            )
            
            # Store active mission
            self.active_missions[mission_id] = {
                "mission": mission,
                "status": MissionStatus.PENDING,
                "created_at": datetime.utcnow(),
                "deadline": deadline,
                "steps_completed": [],
                "steps_failed": [],
                "results": {},
                "execution_log": [],
            }
            
            logger.info(f"Created mission {mission_id}: {goal[:50]}...")
            
            return {
                "success": True,
                "mission_id": mission_id,
                "goal": goal,
                "status": MissionStatus.PENDING.value,
                "priority": priority,
                "deadline": deadline.isoformat() if deadline else None,
            }
            
        except Exception as e:
            logger.error(f"Failed to create mission: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def execute_mission(
        self,
        mission_id: str,
        synchronous: bool = False,
        timeout: int = 3600,  # 1 hour default
    ) -> Dict[str, Any]:
        """Execute a mission.
        
        Args:
            mission_id: Mission identifier
            synchronous: Whether to wait for completion
            timeout: Maximum execution time in seconds
            
        Returns:
            Mission execution result
        """
        try:
            # Check if mission exists
            if mission_id not in self.active_missions:
                return {
                    "success": False,
                    "error": f"Mission {mission_id} not found",
                }
            
            mission_data = self.active_missions[mission_id]
            
            # Check if mission is already running
            if mission_data["status"] != MissionStatus.PENDING:
                return {
                    "success": False,
                    "error": f"Mission {mission_id} is not in pending state",
                    "status": mission_data["status"].value,
                }
            
            # Mark as running
            mission_data["status"] = MissionStatus.RUNNING
            mission_data["started_at"] = datetime.utcnow()
            
            logger.info(f"Starting execution of mission {mission_id}")
            
            if synchronous:
                # Execute synchronously
                return self._execute_mission_sync(mission_id, timeout)
            else:
                # Execute asynchronously
                self._execute_mission_async(mission_id)
                return {
                    "success": True,
                    "mission_id": mission_id,
                    "status": MissionStatus.RUNNING.value,
                    "message": "Mission started",
                }
                
        except Exception as e:
            logger.error(f"Failed to execute mission {mission_id}: {e}")
            if mission_id in self.active_missions:
                self.active_missions[mission_id]["status"] = MissionStatus.FAILED
            
            return {
                "success": False,
                "error": str(e),
            }
    
    def _execute_mission_sync(self, mission_id: str, timeout: int) -> Dict[str, Any]:
        """Execute mission synchronously."""
        start_time = time.time()
        timeout_time = start_time + timeout
        
        # Start async execution
        self._execute_mission_async(mission_id)
        
        # Wait for completion
        while time.time() < timeout_time:
            status = self.get_mission_status(mission_id)
            
            if status["status"] in [MissionStatus.COMPLETED.value, MissionStatus.FAILED.value]:
                return status
            
            time.sleep(5)  # Poll every 5 seconds
        
        # Timeout reached
        self.active_missions[mission_id]["status"] = MissionStatus.FAILED
        self.active_missions[mission_id]["error"] = "Execution timeout"
        
        return {
            "success": False,
            "mission_id": mission_id,
            "status": MissionStatus.FAILED.value,
            "error": "Execution timeout",
            "execution_time": timeout,
        }
    
    def _execute_mission_async(self, mission_id: str):
        """Execute mission asynchronously."""
        try:
            mission_data = self.active_missions[mission_id]
            mission = mission_data["mission"]
            
            # Execute mission using the mission manager
            result = self.mission_manager.execute_mission(mission)
            
            # Update mission data with results
            mission_data["status"] = MissionStatus.COMPLETED if result.success else MissionStatus.FAILED
            mission_data["results"] = result.results
            mission_data["execution_time"] = result.execution_time
            mission_data["completed_steps"] = [step.id for step in result.completed_steps]
            mission_data["failed_steps"] = [step.id for step in result.failed_steps]
            mission_data["finished_at"] = datetime.utcnow()
            
            if not result.success:
                mission_data["error"] = result.error if hasattr(result, 'error') else "Unknown error"
            
            # Move to history
            self.mission_history[mission_id] = mission_data
            del self.active_missions[mission_id]
            
            logger.info(f"Mission {mission_id} completed with status: {mission_data['status'].value}")
            
        except Exception as e:
            logger.error(f"Async mission execution failed: {e}")
            if mission_id in self.active_missions:
                self.active_missions[mission_id]["status"] = MissionStatus.FAILED
                self.active_missions[mission_id]["error"] = str(e)
                self.active_missions[mission_id]["finished_at"] = datetime.utcnow()
    
    def get_mission_status(self, mission_id: str) -> Dict[str, Any]:
        """Get current status of a mission."""
        try:
            # Check active missions
            if mission_id in self.active_missions:
                mission_data = self.active_missions[mission_id]
            # Check mission history
            elif mission_id in self.mission_history:
                mission_data = self.mission_history[mission_id]
            else:
                return {
                    "success": False,
                    "error": f"Mission {mission_id} not found",
                }
            
            # Build status response
            response = {
                "success": True,
                "mission_id": mission_id,
                "status": mission_data["status"].value,
                "created_at": mission_data["created_at"].isoformat(),
            }
            
            if mission_data.get("started_at"):
                response["started_at"] = mission_data["started_at"].isoformat()
            
            if mission_data.get("finished_at"):
                response["finished_at"] = mission_data["finished_at"].isoformat()
            
            if mission_data.get("execution_time"):
                response["execution_time"] = mission_data["execution_time"]
            
            if mission_data.get("completed_steps"):
                response["steps_completed"] = len(mission_data["completed_steps"])
            
            if mission_data.get("failed_steps"):
                response["steps_failed"] = len(mission_data["failed_steps"])
            
            if mission_data.get("error"):
                response["error"] = mission_data["error"]
            
            if mission_data.get("results"):
                response["results"] = mission_data["results"]
            
            if mission_data.get("deadline"):
                response["deadline"] = mission_data["deadline"].isoformat()
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get mission status: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def cancel_mission(self, mission_id: str) -> Dict[str, Any]:
        """Cancel a running mission."""
        try:
            if mission_id not in self.active_missions:
                return {
                    "success": False,
                    "error": f"Mission {mission_id} not found or not active",
                }
            
            mission_data = self.active_missions[mission_id]
            
            if mission_data["status"] != MissionStatus.RUNNING:
                return {
                    "success": False,
                    "error": f"Mission {mission_id} is not running",
                    "status": mission_data["status"].value,
                }
            
            # Mark as cancelled
            mission_data["status"] = MissionStatus.CANCELLED
            mission_data["cancelled_at"] = datetime.utcnow()
            
            # Move to history
            self.mission_history[mission_id] = mission_data
            del self.active_missions[mission_id]
            
            logger.info(f"Cancelled mission {mission_id}")
            
            return {
                "success": True,
                "mission_id": mission_id,
                "status": MissionStatus.CANCELLED.value,
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel mission: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def list_missions(
        self,
        status_filter: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List missions with optional status filter."""
        try:
            all_missions = {
                **self.active_missions,
                **self.mission_history
            }
            
            # Apply status filter
            if status_filter:
                all_missions = {
                    mission_id: mission_data
                    for mission_id, mission_data in all_missions.items()
                    if mission_data["status"].value == status_filter
                }
            
            # Sort by creation time (newest first)
            sorted_missions = sorted(
                all_missions.items(),
                key=lambda x: x[1]["created_at"],
                reverse=True
            )
            
            # Apply limit
            sorted_missions = sorted_missions[:limit]
            
            # Build response
            missions_list = []
            for mission_id, mission_data in sorted_missions:
                mission_info = {
                    "mission_id": mission_id,
                    "goal": mission_data["mission"].goal,
                    "status": mission_data["status"].value,
                    "created_at": mission_data["created_at"].isoformat(),
                }
                
                if mission_data.get("execution_time"):
                    mission_info["execution_time"] = mission_data["execution_time"]
                
                if mission_data.get("steps_completed"):
                    mission_info["steps_completed"] = len(mission_data["steps_completed"])
                
                missions_list.append(mission_info)
            
            return {
                "success": True,
                "missions": missions_list,
                "total_returned": len(missions_list),
                "total_missions": len(self.active_missions) + len(self.mission_history),
            }
            
        except Exception as e:
            logger.error(f"Failed to list missions: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_mission_statistics(self) -> Dict[str, Any]:
        """Get mission execution statistics."""
        try:
            total_missions = len(self.active_missions) + len(self.mission_history)
            
            # Count by status
            status_counts = {}
            for mission_data in {**self.active_missions, **self.mission_history}.values():
                status = mission_data["status"].value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate average execution time
            completed_missions = [
                mission_data for mission_data in self.mission_history.values()
                if mission_data["status"] == MissionStatus.COMPLETED
            ]
            
            avg_execution_time = 0
            if completed_missions:
                total_time = sum(mission_data.get("execution_time", 0) for mission_data in completed_missions)
                avg_execution_time = total_time / len(completed_missions)
            
            return {
                "success": True,
                "total_missions": total_missions,
                "active_missions": len(self.active_missions),
                "status_breakdown": status_counts,
                "average_execution_time": avg_execution_time,
                "success_rate": (
                    len(completed_missions) / max(len(completed_missions) + status_counts.get("failed", 0), 1)
                ) if total_missions > 0 else 0,
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Modal function wrapper for mission execution
def ada_mission(goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Modal mission execution function wrapper."""
    try:
        service = AdaMissionService()
        
        # Create and execute mission in one call
        creation_result = service.create_mission(goal=goal, context=context)
        
        if not creation_result["success"]:
            return creation_result
        
        mission_id = creation_result["mission_id"]
        
        # Execute synchronously with timeout
        execution_result = service.execute_mission(
            mission_id=mission_id,
            synchronous=True,
            timeout=1800,  # 30 minutes timeout
        )
        
        return execution_result
        
    except Exception as e:
        logger.error(f"Modal mission execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "goal": goal,
        }


if __name__ == "__main__":
    # Local testing
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Ada Mission Service")
    parser.add_argument("--goal", required=True, help="Mission goal")
    parser.add_argument("--action", default="execute", choices=["create", "execute", "status", "list"])
    parser.add_argument("--mission-id", help="Mission ID for status/list actions")
    parser.add_argument("--async", action="store_true", help="Execute asynchronously")
    
    args = parser.parse_args()
    
    service = AdaMissionService()
    
    if args.action == "create":
        result = service.create_mission(args.goal)
    elif args.action == "execute":
        result = service.ada_mission(args.goal)
    elif args.action == "status":
        if not args.mission_id:
            result = {"error": "Mission ID required for status check"}
        else:
            result = service.get_mission_status(args.mission_id)
    elif args.action == "list":
        result = service.list_missions()
    
    print(json.dumps(result, indent=2))
