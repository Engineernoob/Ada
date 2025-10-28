"""Hierarchical planning system for decomposing goals into actionable steps."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.settings import get_setting


@dataclass
class PlanStep:
    """Represents a single step in a plan."""
    step_id: str
    tool: str
    action: str
    parameters: Dict[str, Any]
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    completed: bool = False
    result: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    estimated_duration: float = 60.0

    def __post_init__(self) -> None:
        if not self.description:
            self.description = self.parameters.get("description", "")
        elif "description" not in self.parameters and self.description:
            self.parameters = {**self.parameters, "description": self.description}


@dataclass 
class Plan:
    """Represents a complete plan with multiple steps."""
    id: str
    goal: str
    category: str
    priority: float
    confidence: float
    steps: List[PlanStep]
    status: str = "created"  # created, executing, completed, failed, aborted
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    success_rate: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert plan to dictionary representation."""
        return {
            "id": self.id,
            "goal": self.goal,
            "category": self.category,
            "priority": self.priority,
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success_rate": self.success_rate,
            "steps": [
                {
                    "step_id": step.step_id,
                    "tool": step.tool,
                    "action": step.action,
                    "parameters": step.parameters,
                    "dependencies": step.dependencies,
                    "completed": step.completed,
                    "result": step.result,
                    "error": step.error,
                    "timestamp": step.timestamp,
                    "estimated_duration": step.estimated_duration,
                }
                for step in self.steps
            ]
        }
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next available step to execute."""
        for step in self.steps:
            if not step.completed and all(
                self._get_step_by_id(dep).completed for dep in step.dependencies
                if self._get_step_by_id(dep)
            ):
                return step
        return None
    
    def _get_step_by_id(self, step_id: str) -> Optional[PlanStep]:
        """Find a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def update_success_rate(self) -> None:
        """Calculate and update the success rate based on completed steps."""
        if not self.steps:
            self.success_rate = 0.0
            return
        
        completed_steps = sum(1 for step in self.steps if step.completed and step.error is None)
        self.success_rate = completed_steps / len(self.steps)


class Planner:
    """Converts goals into executable plans with decomposed steps."""
    
    def __init__(self) -> None:
        # Plan templates for common goal types
        self.templates = {
            "analyze": self._create_analyze_plan,
            "create": self._create_create_plan, 
            "research": self._create_research_plan,
            "organize": self._create_organize_plan,
            "learn": self._create_learn_plan
        }
        
        # Available tools and their capabilities
        self.tool_capabilities = {
            "file_ops": ["read", "write", "list", "copy", "move"],
            "web_search": ["search", "discover", "find"],
            "summarize": ["summarize", "extract", "condense"],
            "note": ["write", "append", "organize"],
            "analyze": ["analyze", "interpret", "evaluate"]
        }
    
    def plan(self, goal: Goal) -> Optional[Plan]:
        """
        Create a plan for the given goal.
        
        Args:
            goal: The goal to create a plan for
            
        Returns:
            Complete plan or None if planning failed
        """
        goal_type = self._extract_goal_type(goal.goal)
        
        if goal_type not in self.templates:
            # Fallback to generic planning
            return self._create_generic_plan(goal)
        
        try:
            plan = self.templates[goal_type](goal)
            return plan
        except Exception as e:
            print(f"Planning failed for goal '{goal.goal}': {e}")
            return None
    
    def _extract_goal_type(self, goal_text: str) -> str:
        """Extract the main goal type from goal text."""
        goal_lower = goal_text.lower()
        
        for goal_type in ["analyze", "create", "research", "organize", "learn"]:
            if goal_text.startswith(goal_type):
                return goal_type
        
        return "create"  # Default fallback
    
    def _create_analyze_plan(self, goal: Goal) -> Plan:
        """Create a plan for analysis goals."""
        # Extract target from goal
        target = goal.goal.replace("analyze ", "").strip()
        
        steps = [
            PlanStep(
                step_id="locate_target",
                tool="file_ops",
                action="read",
                parameters={"path": target, "description": f"Locate and read {target}"},
                dependencies=[]
            ),
            PlanStep(
                step_id="analyze_content", 
                tool="analyze",
                action="interpret",
                parameters={"input": "locate_target.result", "description": f"Analyze {target} content"},
                dependencies=["locate_target"]
            ),
            PlanStep(
                step_id="summarize_findings",
                tool="summarize", 
                action="run",
                parameters={"input": "analyze_content.result", "description": "Summarize analysis results"},
                dependencies=["analyze_content"]
            ),
            PlanStep(
                step_id="save_summary",
                tool="note",
                action="write", 
                parameters={"text": "summarize_findings.result", "description": "Save analysis summary to journal"},
                dependencies=["summarize_findings"]
            )
        ]
        
        return Plan(
            id=str(uuid.uuid4()),
            goal=goal.goal,
            category=goal.category,
            priority=goal.priority,
            confidence=goal.confidence,
            steps=steps
        )
    
    def _create_research_plan(self, goal: Goal) -> Plan:
        """Create a plan for research goals."""
        # Extract topic from goal
        topic = goal.goal.replace("research ", "").strip()
        
        steps = [
            PlanStep(
                step_id="search_topic",
                tool="web_search",
                action="search",
                parameters={"query": topic, "description": f"Search for information about {topic}"},
                dependencies=[]
            ),
            PlanStep(
                step_id="analyze_sources",
                tool="analyze",
                action="evaluate", 
                parameters={"input": "search_topic.result", "description": "Evaluate search sources"},
                dependencies=["search_topic"]
            ),
            PlanStep(
                step_id="extract_key_info",
                tool="summarize",
                action="extract",
                parameters={"input": "analyze_sources.result", "description": "Extract key information from sources"},
                dependencies=["analyze_sources"]
            ),
            PlanStep(
                step_id="compile_findings",
                tool="note",
                action="write",
                parameters={"text": "extract_key_info.result", "description": "Compile research findings"},
                dependencies=["extract_key_info"]
            )
        ]
        
        return Plan(
            id=str(uuid.uuid4()),
            goal=goal.goal,
            category=goal.category,
            priority=goal.priority,
            confidence=goal.confidence,
            steps=steps
        )
    
    def _create_create_plan(self, goal: Goal) -> Plan:
        """Create a plan for creation goals."""
        # Extract what to create from goal 
        target = goal.goal.replace("create ", "").strip()
        
        steps = [
            PlanStep(
                step_id="analyze_requirements",
                tool="analyze",
                action="evaluate",
                parameters={"input": target, "description": f"Analyze requirements for creating {target}"},
                dependencies=[]
            ),
            PlanStep(
                step_id="create_content",
                tool="file_ops",
                action="write",
                parameters={"path": target, "content": "analyze_requirements.result", "description": f"Create {target}"},
                dependencies=["analyze_requirements"]
            ),
            PlanStep(
                step_id="review_creation",
                tool="analyze",
                action="interpret",
                parameters={"input": "create_content.result", "description": "Review and validate creation"},
                dependencies=["create_content"]
            ),
            PlanStep(
                step_id="log_completion",
                tool="note", 
                action="write",
                parameters={"text": f"Successfully created {target}", "description": "Log creation completion"},
                dependencies=["review_creation"]
            )
        ]
        
        return Plan(
            id=str(uuid.uuid4()),
            goal=goal.goal,
            category=goal.category,
            priority=goal.priority,
            confidence=goal.confidence,
            steps=steps
        )
    
    def _create_organize_plan(self, goal: Goal) -> Plan:
        """Create a plan for organization goals."""
        target = goal.goal.replace("organize ", "").strip()
        
        steps = [
            PlanStep(
                step_id="assess_current_state",
                tool="file_ops",
                action="list",
                parameters={"path": target, "description": f"Assess current state of {target}"},
                dependencies=[]
            ),
            PlanStep(
                step_id="design_organization",
                tool="analyze",
                action="evaluate",
                parameters={"input": "assess_current_state.result", "description": "Design organization structure"},
                dependencies=["assess_current_state"]
            ),
            PlanStep(
                step_id="execute_organization",
                tool="file_ops",
                action="reorganize",
                parameters={"source": target, "plan": "design_organization.result", "description": "Execute reorganization"},
                dependencies=["design_organization"]
            ),
            PlanStep(
                step_id="verify_organization",
                tool="file_ops",
                action="list", 
                parameters={"path": target, "description": "Verify organization results"},
                dependencies=["execute_organization"]
            )
        ]
        
        return Plan(
            id=str(uuid.uuid4()),
            goal=goal.goal,
            category=goal.category,
            priority=goal.priority,
            confidence=goal.confidence,
            steps=steps
        )
    
    def _create_learn_plan(self, goal: Goal) -> Plan:
        """Create a plan for learning goals."""
        topic = goal.goal.replace("learn ", "").strip()
        
        steps = [
            PlanStep(
                step_id="assess_knowledge",
                tool="analyze",
                action="evaluate",
                parameters={"topic": topic, "description": f"Assess current knowledge of {topic}"},
                dependencies=[]
            ),
            PlanStep(
                step_id="find_learning_resources",
                tool="web_search",
                action="search",
                parameters={"query": f"{topic} tutorial guide", "description": f"Find learning resources for {topic}"},
                dependencies=["assess_knowledge"]
            ),
            PlanStep(
                step_id="study_materials",
                tool="analyze",
                action="interpret",
                parameters={"input": "find_learning_resources.result", "description": "Study and analyze learning materials"},
                dependencies=["find_learning_resources"]
            ),
            PlanStep(
                step_id="summarize_learning",
                tool="note",
                action="write",
                parameters={"text": "study_materials.result", "description": "Summarize what was learned"},
                dependencies=["study_materials"]
            )
        ]
        
        return Plan(
            id=str(uuid.uuid4()),
            goal=goal.goal,
            category=goal.category,
            priority=goal.priority,
            confidence=goal.confidence,
            steps=steps
        )
    
    def _create_generic_plan(self, goal: Goal) -> Plan:
        """Create a generic plan as fallback."""
        steps = [
            PlanStep(
                step_id="understand_goal",
                tool="analyze",
                action="evaluate",
                parameters={"goal": goal.goal, "description": "Understand the requirements"},
                dependencies=[]
            ),
            PlanStep(
                step_id="create_solution",
                tool="note",
                action="write",
                parameters={"text": "understand_goal.result", "description": "Create initial solution"},
                dependencies=["understand_goal"]
            ),
            PlanStep(
                step_id="review_result",
                tool="analyze",
                action="interpret",
                parameters={"input": "create_solution.result", "description": "Review the solution"},
                dependencies=["create_solution"]
            )
        ]
        
        return Plan(
            id=str(uuid.uuid4()),
            goal=goal.goal,
            category=goal.category,
            priority=goal.priority,
            steps=steps
        )


# Import Goal class for type hints (placed at the end to avoid circular imports)
from .intent_engine import Goal
