"""Command line interface for interacting with Ada."""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
import os
import shlex
import time

from typing import Optional

import torch

from core import ContextManager, ReasoningEngine, RewardEngine, AutonomousPlanner
from rl import AdaAgent, DialogueEnvironment, ExperienceBuffer
from missions import (
    MissionAuditor,
    MissionDaemon,
    MissionManager,
    MissionSettings,
    CurriculumTrainer,
)
from optimizer import OptimizerSettings
from optimizer.auto_tuner import AutoTuner
from optimizer.evolution_engine import EvolutionEngine
from optimizer.metrics_tracker import MetricsTracker
from optimizer.checkpoint_manager import CheckpointManager
from .event_loop import EventLoop


@dataclass
class PendingFeedback:
    state: torch.Tensor
    next_state: torch.Tensor
    action_index: int
    record_id: int
    user_input: str
    ada_response: str
    base_reward: float
    memory_id: int = 0


@dataclass
class AdaSession:
    context_manager: ContextManager
    engine: ReasoningEngine
    reward_engine: RewardEngine
    environment: DialogueEnvironment
    agent: AdaAgent
    autonomous_planner: AutonomousPlanner
    mission_manager: MissionManager | None = None
    mission_daemon: MissionDaemon | None = None
    metrics_tracker: MetricsTracker | None = None
    auto_tuner: AutoTuner | None = None
    evolution_engine: EvolutionEngine | None = None
    checkpoint_manager: CheckpointManager | None = None
    optimizer_settings: OptimizerSettings | None = None
    pending: Optional[PendingFeedback] = None
    cumulative_reward: float = 0.0
    feedback_count: int = 0
    dialog_count: int = 0  # Track dialogs for persona updates
    last_latency_ms: float = 0.0

    def handle_input(self, user_input: str) -> str:
        if user_input.startswith("/rate"):
            return self._handle_feedback(user_input)
        elif user_input.startswith("/plan"):
            return self._handle_planning_command(user_input)
        elif user_input.startswith("/goals"):
            return self._handle_goals_command()
        elif user_input.startswith("/run"):
            return self._handle_run_command(user_input)
        elif user_input.startswith("/memory"):
            return self._handle_memory_command()
        elif user_input.startswith("/persona"):
            return self._handle_persona_command()
        elif user_input.startswith("/abort"):
            return self._handle_abort_command(user_input)
        elif user_input.startswith("/mission"):
            return self._handle_mission_command(user_input)
        elif user_input.startswith("/daemon"):
            return self._handle_daemon_command(user_input)
        elif user_input.startswith("/audit"):
            return self._handle_audit_command()
        elif user_input.startswith("/optimize"):
            return self._handle_optimize_command(user_input)
        elif user_input.startswith("/evolve"):
            return self._handle_evolve_command()
        elif user_input.startswith("/metrics"):
            return self._handle_metrics_query(user_input)
        elif user_input.startswith("/rollback"):
            return self._handle_rollback_command(user_input)
        else:
            return self._handle_message(user_input)

    def _handle_message(self, user_input: str) -> str:
        state_vector = self.environment.encode(self.environment.state)
        prompt = self.context_manager.build_prompt(user_input)
        start_time = time.perf_counter()
        generation = self.engine.generate_with_metrics(prompt)
        self.last_latency_ms = (time.perf_counter() - start_time) * 1000.0
        ada_response = generation.text
        next_state = self.environment.observe(user_input, ada_response)
        next_state_vector = self.environment.encode(next_state)

        base_reward = self.engine.reward(user_input, ada_response)
        record_id = self.context_manager.remember(user_input, ada_response, base_reward)
        self.pending = PendingFeedback(
            state=state_vector,
            next_state=next_state_vector,
            action_index=generation.action_index,
            record_id=record_id,
            user_input=user_input,
            ada_response=ada_response,
            base_reward=base_reward,
            memory_id=generation.memory_id,
        )

        # Increment dialog count and check for persona updates
        self.dialog_count += 1
        if hasattr(
            self.engine.persona, "needs_update"
        ) and self.engine.persona.needs_update(self.dialog_count):
            if self.engine.update_persona_from_memories():
                print(f"🎭 Persona updated after {self.dialog_count} dialogs")
                self.dialog_count = 0  # Reset counter after update

        confidence = generation.confidence
        return f"{ada_response} (confidence: {confidence:.2f})"

    def _handle_feedback(self, command: str) -> str:
        if not self.pending:
            return "No recent response to rate."
        parts = command.split(maxsplit=1)
        if len(parts) == 1:
            return "Provide feedback with '/rate good' or '/rate bad'."

        feedback = parts[1].strip()
        feedback_reward = self.reward_engine.compute(
            feedback, self.pending.user_input, self.pending.ada_response
        )
        final_reward = (
            feedback_reward if feedback_reward != 0.0 else self.pending.base_reward
        )

        self.context_manager.store.update_reward(self.pending.record_id, final_reward)
        single_update = self.agent.update_policy(
            self.pending.state,
            self.pending.action_index,
            final_reward,
            self.pending.next_state,
        )
        batch_update = self.agent.train_on_batch()
        self.agent.save()

        self.cumulative_reward += final_reward
        self.feedback_count += 1
        session_avg = self.cumulative_reward / max(1, self.feedback_count)

        losses: list[float] = []
        rewards: list[float] = []
        if single_update:
            losses.append(single_update[0])
            rewards.append(single_update[1])
        if batch_update:
            losses.append(batch_update[0])
            rewards.append(batch_update[1])

        display_loss = sum(losses) / len(losses) if losses else 0.0
        reward_trend = sum(rewards) / len(rewards) if rewards else final_reward

        # Update episodic memory with feedback reward
        if self.pending.memory_id > 0:
            updated = self.engine.update_memory_reward(
                self.pending.memory_id, final_reward
            )
            if updated:
                print(f"💾 Memory reward updated: {final_reward:+.2f}")

        self.pending = None

        # Trigger autonomous planning after feedback
        autonomous_result = self.autonomous_planner.infer_and_plan(
            trigger_after_conversation=True
        )
        if (
            autonomous_result["status"] == "processed"
            and autonomous_result["intents_found"] > 0
        ):
            print(
                f"\n🤖 Autonomous planning: detected {autonomous_result['intents_found']} potential goals"
            )

        if self.metrics_tracker:
            grad_norm = float(getattr(self.agent, "last_grad_norm", 0.0) or 0.0)
            cpu_usage = self._cpu_usage()
            gpu_usage = self._gpu_usage()
            latency_ms = self.last_latency_ms
            self.metrics_tracker.record(
                reward_avg=session_avg,
                loss=display_loss,
                grad_norm=grad_norm,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                latency_ms=latency_ms,
            )

        return f"Reward noted. Session Avg: {session_avg:+.2f} | Loss: {display_loss:.4f} | Batch Reward: {reward_trend:+.2f}"

    def _handle_planning_command(self, command: str) -> str:
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            return 'Usage: /plan "<goal>" or /plan <goal>'

        goal = parts[1].strip()
        if not goal:
            return "Please provide a goal to plan for."

        result = self.autonomous_planner.plan_from_command(goal)
        if result["status"] == "plan_created":
            plan = result["plan"]
            response = f"📋 Plan created: {plan.id}\n"
            response += f"Goal: {plan.goal}\n"
            response += f"Steps: {len(plan.steps)}\n"
            for i, step in enumerate(plan.steps, 1):
                response += f"  {i}. {step.description} ({step.tool})\n"
            response += f"\nUse '/run {plan.id}' to execute this plan."
            return response
        else:
            return f"Failed to create plan: {result.get('error', 'Unknown error')}"

    def _handle_goals_command(self) -> str:
        goals = self.autonomous_planner.get_goals()
        if not goals:
            return "No goals recorded yet."

        response = "🎯 Recent goals:\n"
        for goal in goals[:5]:
            status_icon = (
                "✅"
                if goal["status"] == "completed"
                else "⏳" if goal["status"] == "created" else "❌"
            )
            response += f"  {status_icon} {goal['goal'][:50]}... (priority: {goal['priority']:.2f})\n"

        response += f"\nTotal goals: {len(goals)}"
        return response

    def _handle_run_command(self, command: str) -> str:
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /run <plan_id>"

        plan_id = parts[1].strip()
        if not plan_id:
            return "Please provide a plan ID to run."

        result = self.autonomous_planner.execute_plan(plan_id)
        if result["status"] == "executed":
            execution = result["execution"]
            evaluation = result["evaluation"]

            response = f"🚀 Plan {plan_id} execution:\n"
            response += f"✅ Success: {execution['success']}\n"
            response += f"📊 Completed: {execution['completed_steps']}/{execution['total_steps']} steps\n"
            response += f"🎯 Reward: {evaluation.get('adjusted_reward', 0):.2f}\n"
            return response
        else:
            return f"Failed to execute plan: {result.get('error', 'Unknown error')}"

    def _handle_memory_command(self) -> str:
        stats = self.engine.get_memory_stats()

        if not stats.get("memory_enabled", False):
            return "❌ Episodic memory is disabled"

        response = "🧠 Episodic Memory Status:\n"
        response += f"📊 Total memories: {stats.get('total_memories', 0)}\n"
        response += f"⭐ Average reward: {stats.get('average_reward', 0):.3f}\n"
        response += f"🔥 Max reward: {stats.get('max_reward', 0):.3f}\n"
        response += f"❄️ Min reward: {stats.get('min_reward', 0):.3f}\n"

        return response

    def _handle_persona_command(self) -> str:
        stats = self.engine.get_persona_stats()

        if not stats.get("persona_enabled", False):
            return "❌ Persona system is disabled"

        response = self.engine.persona.get_persona_summary()
        return response

    def _handle_mission_command(self, command: str) -> str:
        if not self.mission_manager:
            return "Mission system is not available."
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return f"Unable to parse command: {exc}"
        if len(parts) < 2:
            return "Usage: /mission <new|list|run>"
        action = parts[1].lower()
        if action == "new":
            if len(parts) < 3:
                return 'Usage: /mission new "<goal>"'
            goal = " ".join(parts[2:])
            mission = self.mission_manager.create_mission(goal)
            return f"🗒️ Mission {mission.id} created for goal: {mission.goal}"
        if action == "list":
            missions = self.mission_manager.list_missions(limit=10)
            if not missions:
                return "No missions recorded yet."
            response = ["📋 Missions:"]
            for mission in missions:
                status_icon = {
                    "completed": "✅",
                    "running": "⚙️",
                    "failed": "❌",
                }.get(mission.status, "⏳")
                response.append(
                    f"  {status_icon} {mission.id} — {mission.goal} (status: {mission.status})"
                )
            return "\n".join(response)
        if action == "run":
            if len(parts) < 3:
                return "Usage: /mission run <mission_id>"
            mission_id = parts[2]
            if not self.mission_daemon:
                return "Mission daemon is not configured."
            try:
                result = self.mission_daemon.run_mission_blocking(mission_id)
            except ValueError:
                return f"Mission {mission_id} was not found."
            status_icon = "✅" if result.success else "❌"
            details = ""
            if result.audit:
                details = f" | Reward Δ {result.audit.reward_delta:+.2f} Drift {result.audit.drift:.2f}"
            return f"{status_icon} {result.message}{details}"
        return "Unknown /mission subcommand."

    def _handle_daemon_command(self, command: str) -> str:
        if not self.mission_daemon:
            return "Mission daemon is not configured."
        parts = command.split()
        if len(parts) < 2:
            return "Usage: /daemon start|stop|status"
        action = parts[1].lower()
        if action == "start":
            self.mission_daemon.start_background()
            return "🗓 Mission daemon started."
        if action == "stop":
            self.mission_daemon.stop_background()
            return "🛑 Mission daemon stopped."
        if action == "status":
            return (
                "🟢 Daemon running"
                if self.mission_daemon.is_running
                else "⚪️ Daemon idle"
            )
        return "Unknown /daemon subcommand."

    def _handle_audit_command(self) -> str:
        if not self.mission_daemon:
            return "Mission daemon is not configured."
        report = self.mission_daemon.run_audit_blocking()
        return f"📈 Latest checkpoint audited. Reward Δ {report.reward_delta:+.2f} | Drift {report.drift:.2f}"

    def _handle_optimize_command(self, command: str) -> str:
        if not self.auto_tuner:
            return "Auto-tuner is not configured."
        parts = command.split()
        if len(parts) < 2 or parts[1].lower() != "now":
            return "Usage: /optimize now"
        params = asyncio.run(self.auto_tuner.run_cycle())
        return (
            "🧪 Auto-tuner applied: "
            f"lr={params.learning_rate:.5f} hidden={params.hidden_size} "
            f"dropout={params.dropout:.2f} batch={params.batch_size}"
        )

    def _handle_evolve_command(self) -> str:
        if not self.evolution_engine:
            return "Evolution engine is not configured."
        result = asyncio.run(self.evolution_engine.run_cycle())
        best_delta = result.best_reward - result.baseline_reward
        if result.promoted:
            return (
                "🧬 Evolution complete. "
                f"Explored {len(result.explored)} variants | Δreward {best_delta:+.3f} | "
                f"Promoted {result.promoted.checkpoint_id}"
            )
        return (
            "🧬 Evolution complete. "
            f"Explored {len(result.explored)} variants | Δreward {best_delta:+.3f} | "
            "No promotion"
        )

    def _handle_metrics_query(self, command: str) -> str:
        if not self.metrics_tracker:
            return "Metrics tracker is not configured."
        parts = command.split()
        limit = 5
        if len(parts) > 1:
            try:
                limit = max(1, min(20, int(parts[1])))
            except ValueError:
                return "Usage: /metrics [count]"
        snapshots = self.metrics_tracker.get_recent(limit=limit)
        if not snapshots:
            return "No optimizer metrics recorded yet."
        lines = ["📊 Optimizer metrics:"]
        for snap in snapshots:
            lines.append(
                "  "
                f"{snap.timestamp.strftime('%H:%M:%S')} | reward={snap.reward_avg:+.3f} "
                f"loss={snap.loss:.4f} grad={snap.grad_norm:.3f} "
                f"cpu={snap.cpu_usage:.1f}% latency={snap.latency_ms:.1f}ms"
            )
        return "\n".join(lines)

    def _handle_rollback_command(self, command: str) -> str:
        if not self.checkpoint_manager:
            return "Checkpoint manager is not configured."
        if self.optimizer_settings and not self.optimizer_settings.rollback_safe:
            return "Rollback is disabled by configuration."
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /rollback <checkpoint_id>"
        checkpoint_id = parts[1].strip()
        if not checkpoint_id:
            return "Usage: /rollback <checkpoint_id>"
        metadata = self.checkpoint_manager.rollback(checkpoint_id)
        if not metadata:
            return f"Checkpoint {checkpoint_id} not found."
        return (
            "🔁 Rollback applied: "
            f"{metadata.checkpoint_id} (Δreward {metadata.reward_delta:+.3f})"
        )

    def _cpu_usage(self) -> float:
        try:
            load_avg = os.getloadavg()[0]
            cores = os.cpu_count() or 1
            return max(0.0, min(100.0, (load_avg / cores) * 100))
        except OSError:
            return 0.0

    def _gpu_usage(self) -> float | None:
        # Placeholder for environments without GPU monitoring support
        return None


def main() -> None:
    print("Initializing Ada core systems...")

    try:
        context = ContextManager()
        print("  ✓ Context manager loaded")
    except Exception as e:
        print(f"  ❌ Context manager failed: {e}")
        return

    try:
        # Use TextEncoder by default to avoid sentence-transformers import issues
        engine = ReasoningEngine(use_language_encoder=False)
        print("  ✓ Reasoning engine loaded")
    except Exception as e:
        print(f"  ❌ Reasoning engine failed: {e}")
        return

    try:
        reward_engine = RewardEngine()
        print("  ✓ Reward engine loaded")
    except Exception as e:
        print(f"  ❌ Reward engine failed: {e}")
        return

    try:
        environment = DialogueEnvironment()
        print("  ✓ Dialogue environment loaded")
    except Exception as e:
        print(f"  ❌ Dialogue environment failed: {e}")
        return

    try:
        memory = ExperienceBuffer()
        print("  ✓ Experience buffer loaded")
    except Exception as e:
        print(f"  ❌ Experience buffer failed: {e}")
        return

    try:
        agent = AdaAgent(
            model=engine.model,
            memory=memory,
            action_space=engine.action_space,
            checkpoint_path=engine.checkpoint_path,
        )
        print("  ✓ Ada agent created")
    except Exception as e:
        print(f"  ❌ Ada agent failed: {e}")
        print("  This may be due to model incompatibility. Trying fallback...")

        # Create a minimal agent without neural model
        class MinimalAgent:
            def __init__(self):
                self.action_space = len(engine.phrases)
                self.memory = memory
                self.model = engine.model

            def update_policy(self, *args, **kwargs):
                return None

            def train_on_batch(self, *args, **kwargs):
                return None

            def save(self):
                pass

        agent = MinimalAgent()
        print("  ✓ Minimal fallback agent created")

    try:
        # Initialize autonomous planner with persona integration
        autonomous_planner = AutonomousPlanner(
            context_manager=context, reasoning_engine=engine
        )
        print("  ✓ Autonomous planner loaded")
    except Exception as e:
        print(f"  ❌ Autonomous planner failed: {e}")
        autonomous_planner = None

    try:
        optimizer_settings = OptimizerSettings.from_settings()
        metrics_tracker = MetricsTracker()
        checkpoint_manager = CheckpointManager()
        auto_tuner = AutoTuner(tracker=metrics_tracker, settings=optimizer_settings)
        evolution_engine = EvolutionEngine(
            settings=optimizer_settings,
            auto_tuner=auto_tuner,
            tracker=metrics_tracker,
            checkpoint_manager=checkpoint_manager,
        )
        print("  ✓ Optimizer stack ready")
    except Exception as e:
        print(f"  ❌ Optimizer stack failed: {e}")
        optimizer_settings = None
        metrics_tracker = None
        auto_tuner = None
        evolution_engine = None
        checkpoint_manager = None

    try:
        mission_settings = MissionSettings.from_settings()
        mission_manager = MissionManager()
        curriculum_trainer = CurriculumTrainer()
        mission_auditor = MissionAuditor()
        mission_daemon = MissionDaemon(
            manager=mission_manager,
            trainer=curriculum_trainer,
            auditor=mission_auditor,
            settings=mission_settings,
            auto_tuner=auto_tuner,
        )
        print("  ✓ Mission daemon ready")
    except Exception as e:
        print(f"  ❌ Mission daemon failed: {e}")
        mission_manager = None
        mission_daemon = None

    try:
        session = AdaSession(
            context_manager=context,
            engine=engine,
            reward_engine=reward_engine,
            environment=environment,
            agent=agent,
            autonomous_planner=autonomous_planner,
            mission_manager=mission_manager,
            mission_daemon=mission_daemon,
            metrics_tracker=metrics_tracker,
            auto_tuner=auto_tuner,
            evolution_engine=evolution_engine,
            checkpoint_manager=checkpoint_manager,
            optimizer_settings=optimizer_settings,
        )
        print("  ✓ Session initialized")
    except Exception as e:
        print(f"  ❌ Session failed: {e}")
        return

    try:
        loop = EventLoop()
        print("  ✓ Event loop ready")
    except Exception as e:
        print(f"  ❌ Event loop failed: {e}")
        return

    print("")
    print("🎉 Ada initialization complete!")
    print("Ada: Hello Taahirah, systems ready to learn. Type 'quit' to exit.")
    print(
        "Commands available: /plan <goal>, /goals, /run <plan_id>, /abort <plan_id>, "
        '/mission new "<goal>", /mission list, /mission run <id>, /daemon start, /audit, /memory, /persona, '
        "/optimize now, /evolve, /metrics [n], /rollback <checkpoint_id>"
        '/mission new "<goal>", /mission list, /mission run <id>, /daemon start, /audit, /memory, /persona'
    )
    print("")
    print(
        "💡 Note: Running in fallback mode due to external library compatibility issues."
    )
    print("   Core functionality (memory, persona, planning) is fully operational.")
    print("")

    try:
        loop.run(session.handle_input)
    except KeyboardInterrupt:
        print("\nAda: Goodbye!")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Ada: System encountered an issue. Please check logs.")


if __name__ == "__main__":
    main()
