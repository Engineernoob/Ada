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
            self.executor.register_plan(plan)

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
        self.executor.register_plan(plan)

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
        target_plan = self.executor.get_plan(plan_id)
        if not target_plan:
            return {"status": "error", "error": f"Plan {plan_id} not found or already executed"}

        if target_plan.status != "created":
            return {"status": "error", "error": f"Plan {plan_id} already executed"}

        print(f"🚀 executing plan: {plan_id}")
        execution_result = self.executor.run_plan(target_plan)
        evaluation = self.evaluator.evaluate_plan(plan_id, execution_result)

        return {
            "status": "executed",
            "execution": execution_result,
            "evaluation": evaluation
        }

    def get_goals(self) -> List[Dict[str, Any]]:
        """Get recent discovered goals and their status."""
        plan_summaries = self.executor.get_plan_summaries(limit=10)
        return [
            {
                "id": summary.get("id", ""),
                "goal": summary.get("goal", ""),
                "category": summary.get("category", ""),
                "priority": float(summary.get("priority", 0.0)),
                "confidence": float(summary.get("confidence", 0.0)),
                "status": summary.get("status", "created"),
                "created_at": summary.get("created_at", 0.0),
            }
            for summary in plan_summaries
        ]

    def get_performance_report(self) -> str:
        """Get a performance report of autonomous actions."""
        return self.evaluator.get_evaluation_report(limit=5)

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights for improving future autonomous actions."""
        return self.evaluator.get_learning_insights()
