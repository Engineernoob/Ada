"""Integration layer for autonomous planning in Ada's event loops."""

from __future__ import annotations

from typing import Any, Dict, List

from .context_manager import ContextManager
from agent import Evaluator, Executor
from planner import Planner, IntentEngine


class AutonomousPlanner:
    def __init__(self, context_manager: ContextManager, reasoning_engine=None) -> None:
        self.context_manager = context_manager
        self.reasoning_engine = reasoning_engine
        self.intent_engine = IntentEngine()
        self.planner = Planner()
        self.executor = Executor()
        self.evaluator = Evaluator()
        
    def infer_and_plan(self, trigger_after_conversation: bool = True) -> Dict[str, Any]:
        """
        Infer intents and create autonomous plans after conversations.
        Returns execution results or inference status.
        """
        # Get conversation context
        conversation_history = self.context_manager.build_prompt("")
        
        if not conversation_history or not trigger_after_conversation:
            return {"status": "no_action", "reason": "insufficient context or disabled"}
        
        # Infer intents
        intents = self.intent_engine.infer(conversation_history)
        
        # If reasoning engine has persona, use persona-aligned goal selection
        if self.reasoning_engine and hasattr(self.reasoning_engine, 'persona') and self.reasoning_engine.persona:
            persona_summary = self.reasoning_engine.persona.get_persona_summary()
            persona_intents = self.intent_engine.infer_from_persona(persona_summary, conversation_history[-1] if conversation_history else "")
            intents.extend(persona_intents)
        
        if not intents:
            return {"status": "no_intents", "reason": "no clear goals detected"}
        
        results = []
        for intent in intents[:2]:  # Limit to top 2 intents to avoid overload
            # Create plan for each intent
            plan = self.planner.plan(intent)
            
            # Log plan creation
            # Note: Using the existing logger interface
            # self.executor.logger.log_plan_creation(
            #     plan.id, plan.goal, plan.category, plan.priority, plan.confidence
            # )
            
            print(f"🧭 Goal inferred: {plan.goal} (confidence: {plan.confidence:.2f})")
            
            # Execute plan if high enough confidence and priority
            if plan.confidence > 0.5 and plan.priority > 0.6:
                print(f"🚀 Executing plan: {plan.id}")
                execution_result = self.executor.run_plan(plan)
                
                # Evaluate the execution
                evaluation = self.evaluator.evaluate_plan(plan.id, execution_result)
                
                results.append({
                    "intent": intent,
                    "plan": plan,
                    "execution": execution_result,
                    "evaluation": evaluation
                })
                
                # Log results as conversation
                outcome_text = f"Executed autonomous plan: {plan.goal}. "
                if execution_result["success"]:
                    outcome_text += f"✅ Success with reward {evaluation.get('adjusted_reward', 0):.2f}"
                else:
                    outcome_text += f"❌ Failed: {execution_result.get('error', 'Unknown error')}"
                
                self.context_manager.remember("Autonomous action", outcome_text, 0)
            else:
                print(f"⏸️ Plan confidence ({plan.confidence:.2f}) or priority ({plan.priority:.2f}) too low for automatic execution")
                results.append({
                    "intent": intent,
                    "plan": plan,
                    "execution": None,
                    "evaluation": None
                })
        
        return {
            "status": "processed",
            "intents_found": len(intents),
            "results": results
        }

    def plan_from_command(self, goal: str, category: str = "action") -> Dict[str, Any]:
        """
        Create and execute a plan from an explicit user command.
        """
        # Create synthetic intent
        from planner import Intent
        intent = Intent(
            goal=goal,
            priority=0.8,
            category=category,
            confidence=0.9
        )
        
        # Create plan
        plan = self.planner.plan(intent)
        
        print(f"🎯 Creating plan for: {goal}")
        print(f"   Category: {category}")
        print(f"   Steps: {len(plan.steps)}")
        
        # Log plan creation
        # Note: Using the existing logger interface
        # self.executor.logger.log_plan_creation(
        #     plan.id, plan.goal, plan.category, plan.priority, plan.confidence
        # )
        
        return {
            "plan": plan,
            "intent": intent,
            "status": "plan_created"
        }

    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a previously created plan by ID."""
        # This would involve retrieving the plan from storage
        # For now, we'll implement a simple lookup
        plan_records = list(self.executor.logger.get_plans(limit=100))
        
        target_plan = None
        for record in plan_records:
            if record.id == plan_id and record.status == "created":
                # Reconstruct plan from record and actions
                # This is simplified - in a full implementation we'd store the full plan JSON
                target_plan = self._reconstruct_plan(record)
                break
        
        if not target_plan:
            return {"status": "error", "error": f"Plan {plan_id} not found or already executed"}
        
        print(f"🚀 executing plan: {plan_id}")
        execution_result = self.executor.run_plan(target_plan)
        evaluation = self.evaluator.evaluate_plan(plan_id, execution_result)
        
        return {
            "status": "executed",
            "execution": execution_result,
            "evaluation": evaluation
        }

    def _reconstruct_plan(self, record) -> Any:
        """Reconstruct a plan object from a record."""
        from planner.planner import Plan, PlanningStep
        
        # Get the actions for this plan
        actions = list(self.executor.logger.get_plan_actions(record.id))
        
        # Reconstruct steps
        steps = []
        for action in actions:
            step = PlanningStep(
                tool=action.tool,
                description=action.description,
                input_data=action.input_data,
                estimated_duration=action.duration_seconds
            )
            steps.append(step)
        
        return Plan(
            goal=record.goal,
            steps=steps,
            priority=record.priority,
            confidence=record.confidence,
            category=record.category,
            id=record.id
        )

    def get_goals(self) -> List[Dict[str, Any]]:
        """Get recent discovered goals and their status."""
        plan_records = list(self.executor.logger.get_plans(limit=10))
        
        goals = []
        for record in plan_records:
            goals.append({
                "id": record.id,
                "goal": record.goal,
                "category": record.category,
                "priority": record.priority,
                "confidence": record.confidence,
                "status": record.status,
                "created_at": record.created_at
            })
        
        return goals

    def get_performance_report(self) -> str:
        """Get a performance report of autonomous actions."""
        return self.evaluator.get_evaluation_report(limit=5)

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights for improving future autonomous actions."""
        return self.evaluator.get_learning_insights()
