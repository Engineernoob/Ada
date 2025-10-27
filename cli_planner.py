#!/usr/bin/env python3
"""Standalone CLI for Ada's autonomous planning functionality."""

import argparse
from pathlib import Path
from typing import Optional

from core import ContextManager, AutonomousPlanner
from agent import Executor, Evaluator
from planner import ActionPlanner, IntentEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Ada autonomous planning CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Create and optionally execute a plan")
    plan_parser.add_argument("goal", help="Goal to create a plan for")
    plan_parser.add_argument("--category", "-c", default="action", 
                            choices=["research", "action", "clarification"],
                            help="Category of the goal")
    plan_parser.add_argument("--execute", "-e", action="store_true",
                            help="Execute the plan after creation")

    # Run command
    run_parser = subparsers.add_parser("run", help="Execute a plan by ID")
    run_parser.add_argument("plan_id", help="ID of the plan to execute")

    # Goals command
    goals_parser = subparsers.add_parser("goals", help="Show recent goals")
    goals_parser.add_argument("--limit", "-l", type=int, default=10,
                             help="Number of goals to show")

    # Performance command
    perf_parser = subparsers.add_parser("performance", help="Show performance report")
    perf_parser.add_argument("--insights", "-i", action="store_true",
                           help="Show learning insights")

    # Tools command
    tools_parser = subparsers.add_parser("tools", help="Show available tools")
    tools_parser.add_argument("--test", "-t", help="Test a specific tool")
    tools_parser.add_argument("--input", help="Input for tool test")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize components
    context = ContextManager()
    autonomous_planner = AutonomousPlanner(context_manager=context)

    try:
        if args.command == "plan":
            result = autonomous_planner.plan_from_command(args.goal, args.category)
            
            if result["status"] == "plan_created":
                plan = result["plan"]
                print(f"📋 Plan created: {plan.id}")
                print(f"🎯 Goal: {plan.goal}")
                print(f"📂 Category: {plan.category}")
                print(f"⚡ Priority: {plan.priority:.2f}, Confidence: {plan.confidence:.2f}")
                print(f"📝 Steps ({len(plan.steps)}):")
                
                for i, step in enumerate(plan.steps, 1):
                    duration = step.estimated_duration
                    print(f"  {i}. {step.description}")
                    print(f"     Tool: {step.tool} (~{duration}s)")
                
                if args.execute:
                    print(f"\n🚀 Executing plan {plan.id}...")
                    exec_result = autonomous_planner.execute_plan(plan.id)
                    
                    if exec_result["status"] == "executed":
                        execution = exec_result["execution"]
                        evaluation = exec_result["evaluation"]
                        
                        print(f"✅ Success: {execution['success']}")
                        print(f"📊 Completed: {execution['completed_steps']}/{execution['total_steps']} steps")
                        print(f"🎯 Reward: {evaluation.get('adjusted_reward', 0):.2f}")
                    else:
                        print(f"❌ Failed: {exec_result.get('error', 'Unknown error')}")
                else:
                    print(f"\n💡 To execute: python cli_planner.py run {plan.id}")
            else:
                print(f"❌ Failed to create plan: {result.get('error', 'Unknown error')}")

        elif args.command == "run":
            result = autonomous_planner.execute_plan(args.plan_id)
            
            if result["status"] == "executed":
                execution = result["execution"]
                evaluation = result["evaluation"]
                
                print(f"🚀 Plan {args.plan_id} executed:")
                print(f"✅ Success: {execution['success']}")
                print(f"📊 Completed: {execution['completed_steps']}/{execution['total_steps']} steps")
                print(f"🎯 Reward: {evaluation.get('adjusted_reward', 0):.2f}")
                
                if execution.get("results"):
                    print("\n📝 Step results:")
                    for step_result in execution["results"]:
                        if step_result["success"]:
                            print(f"  ✅ Step {step_result['step_number']}: {step_result['description'][:60]}...")
                        else:
                            print(f"  ❌ Step {step_result['step_number']}: {step_result.get('error', 'Unknown error')}")
            else:
                print(f"❌ Failed to execute plan: {result.get('error', 'Unknown error')}")

        elif args.command == "goals":
            goals = autonomous_planner.get_goals()
            limited_goals = goals[:args.limit]
            
            if not limited_goals:
                print("📭 No goals recorded yet.")
            else:
                print(f"🎯 Recent {len(limited_goals)} goals (showing {args.limit} of {len(goals)}):")
                for goal in limited_goals:
                    status_icon = {"completed": "✅", "created": "⏳", "failed": "❌"}.get(goal["status"], "❓")
                    print(f"  {status_icon} {goal['goal'][:60]}...")
                    print(f"     📂 {goal['category']} | ⚡ {goal['priority']:.2f} | 🆔 {goal['id']}")

        elif args.command == "performance":
            print(autonomous_planner.get_performance_report())
            
            if args.insights:
                insights = autonomous_planner.get_learning_insights()
                print("\n🧠 Learning Insights:")
                if "message" in insights:
                    print(f"  {insights['message']}")
                else:
                    print(f"  📈 Total evaluations: {insights.get('total_evaluations', 0)}")
                    print(f"  📊 Average score: {insights.get('average_score', 0):.2f}")
                    
                    success_patterns = insights.get("success_patterns", [])
                    if success_patterns:
                        print(f"  ✅ Success patterns: {', '.join(success_patterns[:5])}")
                    
                    failure_patterns = insights.get("failure_patterns", [])
                    if failure_patterns:
                        print(f"  ❌ Failure patterns: {', '.join(failure_patterns[:5])}")
                    
                    recommendations = insights.get("recommendations", [])
                    if recommendations:
                        print(f"  💡 Recommendations:")
                        for rec in recommendations:
                            print(f"    • {rec}")

        elif args.command == "tools":
            from tools.registry import list_available_tools, execute_tool
            
            tools = list_available_tools()
            print(f"🔧 Available tools ({len(tools)}):")
            for tool in tools:
                print(f"  • {tool}")
            
            if args.test:
                if args.test in tools:
                    print(f"\n🧪 Testing tool: {args.test}")
                    try:
                        result = execute_tool(args.test, args.input or "test")
                        print(f"✅ Result: {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")
                    except Exception as e:
                        print(f"❌ Error: {str(e)}")
                else:
                    print(f"❌ Tool '{args.test}' not found")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")


if __name__ == "__main__":
    main()
