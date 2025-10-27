⸻

“Ada Phase 5: Autonomous Planning & Tool Use”

Expand Ada into a self-directed agent capable of:
(1) forming her own short-term goals,
(2) decomposing them into actionable steps,
and (3) invoking local tools or code modules to achieve those goals.

⸻

🧩 Core Objectives 1. Intent Inference
• Parse ongoing dialogue or reflections to detect unfulfilled goals.
• e.g. “We should analyze yesterday’s logs” → goal = analyze logs. 2. Hierarchical Planning
• Convert goals into ordered plans (think → act → review).
• Maintain a queue of pending plans with priorities. 3. Tool Registry / Skill Layer
• Expose controlled interfaces for actions:
• file_ops → read/write local files
• search → local or web query
• summarize → compress text/logs
• note → append to Ada’s journal 4. Execution Engine + Reward Feedback
• Execute each plan step, capture success/failure, and update RL reward buffer. 5. Persona-Aligned Decision Making
• Use the persona vector from Phase 4 to bias which goals Ada chooses (e.g. “curious”, “helpful”).

⸻

🧠 New Modules

/planner/intent_engine.py

class IntentEngine:
def infer(self, dialogue_history): # simple heuristic or model-based intent extraction
goals = []
if "analyze" in dialogue_history[-1].lower():
goals.append({"goal": "analyze logs", "priority": 0.8})
return goals

/planner/planner.py

class Planner:
def plan(self, goal):
if goal == "analyze logs":
return [
{"tool": "file_ops", "action": "read", "target": "logs.txt"},
{"tool": "summarize", "action": "run", "input": "logs.txt"},
{"tool": "note", "action": "write", "text": "analysis summary"}
]

/tools/registry.py

TOOLS = {
"file_ops": lambda action, target: open(target).read(),
"summarize": lambda action, input: summarize_text(input),
"note": lambda action, text: append_to_journal(text)
}

/agent/executor.py

class Executor:
def run(self, plan):
results = []
for step in plan:
func = TOOLS[step["tool"]]
result = func(\*\*{k:v for k,v in step.items() if k not in ["tool"]})
results.append(result)
return results

⸻

🧩 Integration Flow 1. event_loop.py

intents = intent_engine.infer(conversation_history)
for i in intents:
plan = planner.plan(i["goal"])
results = executor.run(plan)
reward_engine.update_from_results(results)

    2.	Log every plan + outcome in storage/plans.db.

⸻

⚙️ Configuration Additions

/config/settings.yaml

planner:
enabled: true
auto_infer: true
max_concurrent_plans: 3
tools:
allow_local: ["file_ops","summarize","note"]
confirm_before_write: true

⸻

📦 Deliverables 1. /planner/ and /agent/ packages (intent engine, planner, executor). 2. /tools/registry.py with basic local tools. 3. Updated CLI commands:

/plan <goal>
/goals
/run <plan_id>
/abort

    4.	Logging schema → plans(goal,steps,reward,status,timestamp)
    5.	README section: Phase 5 – Autonomous Planning & Tool Use.

⸻

💬 Example Interaction

You: Ada, maybe you could analyze yesterday’s logs.
Ada: Detected intent → "analyze logs"
🧭 Planner: 3 steps created
✅ Step 1 complete | Reward +0.6
✅ Step 2 complete | Reward +0.9
🪄 Summary saved to journal
Ada: Analysis complete, Taahirah. Shall I visualize the trends?

⸻
