"""Action outcome evaluation module for Ada's learning system."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from .storage import ActionLogger


class Evaluator:
    def __init__(self, logger: ActionLogger | None = None) -> None:
        self.logger = logger or ActionLogger()
        self.evaluation_history: List[Dict[str, Any]] = []

    def assess(
        self, goal: str, result: Dict[str, Any], context: str | None = None
    ) -> float:
        """
        Evaluate the outcome of an action or plan against its goal.
        Returns a reward score between -1.0 (failure) and 1.0 (success).
        """
        if not isinstance(result, dict):
            return -0.5

        base_score = self._calculate_base_score(goal, result)
        context_modifier = self._calculate_context_modifier(goal, context)
        quality_modifier = self._calculate_quality_modifier(result)

        final_score = base_score + context_modifier + quality_modifier
        final_score = max(-1.0, min(1.0, final_score))

        evaluation_record = {
            "goal": goal,
            "result_summary": result.get("output", "")[:100],
            "context": context,
            "base_score": base_score,
            "context_modifier": context_modifier,
            "quality_modifier": quality_modifier,
            "final_score": final_score,
            "timestamp": datetime.now().isoformat(),
        }

        self.evaluation_history.append(evaluation_record)
        self.logger.log_evaluation(evaluation_record)

        return final_score

    def _calculate_base_score(self, goal: str, result: Dict[str, Any]) -> float:
        """Calculate the base success score from result metrics."""
        if result.get("success", False):
            # Success based on completion rate
            completed = result.get("completed_steps", 1)
            total = result.get("total_steps", 1)
            completion_rate = completed / max(1, total)

            # Base score is the completion rate
            base_score = 0.5 + 0.5 * completion_rate
        else:
            # Failed execution
            error_msg = (result.get("error") or "").lower()

            # Some failures are less severe than others
            if "timeout" in error_msg:
                base_score = -0.3
            elif "not found" in error_msg or "missing" in error_msg:
                base_score = -0.5
            elif "permission" in error_msg or "access denied" in error_msg:
                base_score = -0.8
            else:
                base_score = -0.6

        return base_score

    def _calculate_context_modifier(self, goal: str, context: str | None) -> float:
        """Calculate modifier based on context and goal alignment."""
        if not context:
            return 0.0

        context_lower = context.lower()
        goal_lower = goal.lower()

        modifier = 0.0

        # Positive modifiers for context-goal alignment
        if any(
            word in context_lower
            for word in ["user asked", "user wants", "explicit request"]
        ):
            modifier += 0.1

        if any(word in goal_lower for word in ["important", "urgent", "critical"]):
            modifier += 0.1

        if "follow-up" in context_lower and any(
            word in goal_lower for word in ["research", "learn", "explore"]
        ):
            modifier += 0.2

        # Negative modifiers for context mismatches
        if "low priority" in context_lower:
            modifier -= 0.1

        if "experiment" in context_lower or "test" in context_lower:
            modifier -= 0.05  # Less penalty for experimental actions

        return modifier

    def _calculate_quality_modifier(self, result: Dict[str, Any]) -> float:
        """Calculate modifier based on output quality and efficiency."""
        modifier = 0.0

        # Duration-based modifiers
        duration = result.get("duration_seconds", 0)
        if duration > 0 and duration < 10:
            modifier += 0.1  # Quick execution
        elif duration > 300:  # 5 minutes
            modifier -= 0.1  # Too slow

        # Output quality modifiers
        output = result.get("output", "")
        if output:
            output_length = len(output)

            # Penalize very short or empty outputs
            if output_length < 10:
                modifier -= 0.1
            # Reward detailed outputs
            elif output_length > 500:
                modifier += 0.1

            # Content quality indicators
            output_lower = output.lower()
            if "error" in output_lower or "failed" in output_lower:
                modifier -= 0.2
            elif any(
                word in output_lower
                for word in ["success", "completed", "found", "result"]
            ):
                modifier += 0.1

        return modifier

    def evaluate_plan(
        self, plan_id: str, execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of an entire plan execution."""
        goal = execution_result.get("goal", "Unknown goal")
        success = execution_result.get("success", False)
        completed_steps = execution_result.get("completed_steps", 0)
        total_steps = execution_result.get("total_steps", 1)
        steps = execution_result.get("results", [])

        base_reward = self.assess(goal, execution_result)

        # Calculate step-specific evaluations
        step_evaluations = []
        for step_result in steps:
            step_reward = self.assess(
                step_result.get("description", "Step execution"), step_result
            )
            step_evaluations.append(
                {
                    "step": step_result.get("step_number", 0),
                    "tool": step_result.get("tool", "unknown"),
                    "reward": step_reward,
                }
            )

        # Consistency check: step rewards should align with overall reward
        if step_evaluations:
            avg_step_reward = sum(se["reward"] for se in step_evaluations) / len(
                step_evaluations
            )
            consistency_score = 1.0 - abs(base_reward - avg_step_reward)
        else:
            consistency_score = 0.5

        plan_evaluation = {
            "plan_id": plan_id,
            "goal": goal,
            "overall_success": success,
            "completion_rate": completed_steps / max(1, total_steps),
            "base_reward": base_reward,
            "step_evaluations": step_evaluations,
            "consistency_score": consistency_score,
            "adjusted_reward": base_reward * consistency_score,
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.log_plan_evaluation(plan_evaluation)
        return plan_evaluation

    def get_evaluation_report(self, limit: int = 5) -> str:
        """Generate a human-readable report of recent evaluations."""
        recent_evaluations = self.evaluation_history[-limit:]

        if not recent_evaluations:
            return "No evaluations recorded yet."

        report_lines = [
            f"Recent Evaluation Report ({len(recent_evaluations)} most recent):"
        ]
        report_lines.append("=" * 50)

        for i, eval_record in enumerate(recent_evaluations, 1):
            score = eval_record["final_score"]
            emoji = "🟢" if score > 0.5 else "🟡" if score > 0 else "🔴"

            report_lines.append(f"\n{i}. {emoji} Score: {score:.2f}")
            report_lines.append(f"   Goal: {eval_record['goal'][:60]}...")

            if eval_record.get("result_summary"):
                report_lines.append(f"   Result: {eval_record['result_summary']}...")

            report_lines.append(f"   Time: {eval_record['timestamp']}")

        # Summary statistics
        scores = [e["final_score"] for e in recent_evaluations]
        if scores:
            avg_score = sum(scores) / len(scores)
            report_lines.append(f"\n📊 Average Score: {avg_score:.2f}")
            report_lines.append(
                f"📈 Success Rate: {sum(1 for s in scores if s > 0) / len(scores):.1%}"
            )

        return "\n".join(report_lines)

    def get_learning_insights(self) -> Dict[str, Any]:
        """Generate insights for improving future action selection."""
        if len(self.evaluation_history) < 3:
            return {"message": "Insufficient data for insights"}

        # Analyze patterns
        high_score_evals = [
            e for e in self.evaluation_history if e["final_score"] > 0.5
        ]
        low_score_evals = [e for e in self.evaluation_history if e["final_score"] < 0]

        # Extract common themes
        high_score_contexts = set()
        low_score_contexts = set()

        for eval_record in high_score_evals:
            context = eval_record.get("context", "")
            if context:
                words = [w.strip().lower() for w in context.split() if len(w) > 3]
                high_score_contexts.update(words)

        for eval_record in low_score_evals:
            context = eval_record.get("context", "")
            if context:
                words = [w.strip().lower() for w in context.split() if len(w) > 3]
                low_score_contexts.update(words)

        insights = {
            "total_evaluations": len(self.evaluation_history),
            "average_score": sum(e["final_score"] for e in self.evaluation_history)
            / len(self.evaluation_history),
            "success_patterns": list(high_score_contexts - low_score_contexts)[:10],
            "failure_patterns": list(low_score_contexts - high_score_contexts)[:10],
            "recommendations": (
                [
                    "Focus on actions with clear user requests",
                    "Prefer quick-execution tools for time-sensitive goals",
                    "Ensure input validation for external API calls",
                ]
                if sum(e["final_score"] for e in self.evaluation_history)
                / len(self.evaluation_history)
                < 0.3
                else []
            ),
        }

        return insights
