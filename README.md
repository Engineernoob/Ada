# Ada Scaffold (Phases 1-5 + Phase 3 Fix)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
```

## Usage

```bash
make run
```

## Interactive Reinforcement Learning

- Launch Ada: `make run`
- After each response, provide feedback in the CLI:
  - `/rate good` or `/rate +1` to reward
  - `/rate bad` or `/rate -1` to penalize
- View rolling metrics echoed after each rating.

### Phase 5 - Autonomous Exploration & Tool-Use Agent

Ada now includes autonomous planning capabilities with the following new features:

#### Autonomous Planning Commands
- `/plan "<goal>"` - Create a plan for a specific goal
- `/goals` - Show recent autonomous goals and their status
- `/run <plan_id>` - Execute a previously created plan

#### Standalone Planning CLI
```bash
python3 cli_planner.py plan "research topic" --category research
python3 cli_planner.py plan "create summary" --execute
python3 cli_planner.py goals --limit 10
python3 cli_planner.py performance --insights
python3 cli_planner.py tools --test web_search --input "query"
```

#### Available Tools
- **web_search** - Search for information (uses mdfind on macOS)
- **summarize** - Summarize text content and save to storage
- **note** - Save notes with metadata
- **analyze** - Alias for web_search
- **extract** - Alias for summarize
- **create** - Alias for note

### Phase 3.5 - Conversational Memory & Semantic Recall

Ada now includes episodic memory capabilities with the following features:

#### Memory Commands
- `/memory` - Show memory statistics and status

#### Memory Features
- Stores conversation exchanges with embeddings for semantic retrieval
- Retrieves relevant past conversations to provide context
- Automatic memory integration with reasoning engine
- Persistent storage with SQLite database

### Phase 4 - Persona Formation & Identity Consolidation

Ada now develops and maintains a consistent personality through:

#### Persona Commands
- `/persona` - Display current persona statistics and characteristics

#### Persona Features
- **Persona Vector Generation** - Aggregates patterns from Ada's responses into persistent embeddings
- **Style Bias Application** - Biases response generation toward Ada's learned voice and tone  
- **Drift Detection** - Monitors personality changes and warns when recalibration is needed
- **Automatic Updates** - Persona updates after configured intervals (default: 20 dialogs)
- **Self-Description** - Ada can describe her own personality traits in plain language

#### Example Persona Output
```
Ada Persona Summary:
Tone: warm, confident
Phrasing: concise, polite
Drift: 0.08 since last update
```

### Phase 5.5 - Autonomous Planning & Tool Use

Ada now operates as a self-directed agent capable of forming and executing her own goals:

#### Planning Commands
- `/plan "<goal>"` - Create an autonomous plan for a specific goal
- `/goals` - Show recent inferred goals and their planning status
- `/run <plan_id>` - Execute a previously created plan
- `/abort <plan_id>` - Abort a running plan

#### Autonomous Planning Features
- **Intent Inference** - Automatically detects goals from ongoing dialogue and reflections
- **Hierarchical Planning** - Decomposes goals into ordered, executable steps
- **Tool Registry** - Provides controlled interfaces for file operations, web search, summarization, and note-taking
- **Execution Engine** - Executes plans with success/failure tracking and reward feedback
- **Persona-Aligned Decision Making** - Uses persona vector to bias goal selection (curious, helpful, analytical)

#### Available Tools
- **file_ops** - Read/write/list local files and directories
- **web_search** - Local and web information discovery
- **summarize** - Content summarization and key information extraction  
- **note** - Journal operations and note-taking
- **analyze** - Data interpretation and evaluation

#### Example Interaction
```
You: Ada, maybe you could analyze yesterday's logs.
Ada: Detected intent → "analyze logs"
🧭 Planner: 3 steps created
✅ Step 1 complete | Reward +0.6
✅ Step 2 complete | Reward +0.9
📝 Summary saved to journal
Ada: Analysis complete, Taahirah. Shall I visualize the trends?
```

#### Goal Categories
- **research** - Learn and explore topics
- **action** - Create, build, write, organize
- **clarification** - Ask questions and seek understanding

### Phase 7 - Mission Daemon & Continuous Learning

Ada now runs long-horizon improvement routines via the Mission Daemon:

#### Mission Commands
- `/mission new "<goal>"` - Capture a new long-running goal
- `/mission list` - Review recent missions and their status
- `/mission run <mission_id>` - Execute a mission immediately in the foreground

#### Daemon & Audit Commands
- `/daemon start` - Launch the asynchronous mission daemon thread
- `/daemon stop` - Stop the background daemon
- `/daemon status` - Check daemon health
- `/audit` - Force a checkpoint audit and log reward/drift deltas

#### Mission Runtime Behavior
- Missions are stored in `storage/missions.db`
- Background logs persist to `storage/logs/mission.log`
- Checkpoints are tracked under `storage/checkpoints/`
- Audit metadata is saved to `storage/checkpoints/mission_audit.json`

#### Manual Daemon Startup
```bash
make run  # launch CLI
# inside the CLI
/daemon start
```

The daemon wakes every 60 minutes (configurable in `config/settings.yaml`) to execute pending missions, fine-tune models, and audit checkpoints.

### Phase 8 - Self-Optimization & Neural Evolution

Ada now includes an optimizer stack that tunes hyperparameters, explores model variants, and preserves the best checkpoints.

#### Optimizer Commands
- `/optimize now` - Run the auto-tuner against recent metrics and apply new hyperparameters
- `/evolve` - Launch a genetic-style search over model variants and promote the best candidate
- `/metrics [n]` - Display the latest `n` optimizer metric snapshots (defaults to 5)
- `/rollback <checkpoint_id>` - Restore a promoted checkpoint (requires `optimizer.rollback_safe: true`)

#### Optimizer Features
- **Metrics Tracker** - Records reward, loss, gradient norm, CPU usage, and latency in `storage/optimizer/metrics.db` and logs to `storage/logs/optimizer.log`
- **Auto-Tuner** - Adjusts learning rate, hidden size, dropout, activation, and batch size using recorded metrics (`storage/optimizer/hyperparams.json`)
- **Evolution Engine** - Generates and evaluates candidate configurations, storing experiment history in `storage/optimizer/experiments.db`
- **Checkpoint Manager** - Saves top-performing checkpoints to `storage/checkpoints/` with metadata tracked in `storage/optimizer/checkpoints.json`

The mission daemon automatically triggers the auto-tuner after optimization-focused missions, keeping Ada's models aligned with long-running goals.

## Offline Training

```bash
make train
```

### Replay Fine-Tuning

```bash
make train-rl
```

Runs Ada's adaptive loop over stored conversations and persists updated weights.

## Running Ada in Voice-Learning Mode

1. Ensure local binaries for Whisper.cpp and Piper are installed and paths configured in `config/settings.yaml`.
2. Connect a microphone compatible with macOS CoreAudio.
3. Start the voice loop:

```bash
make run-voice
```

Real-time HUD will display listening/processing states and tone-derived rewards. Ada automatically speaks responses using the configured voice model.

## Notes

- Install Whisper.cpp, Piper, and Silero VAD assets locally for full voice functionality.
- Run `make mps-test` to verify Apple Silicon acceleration.
- Phase 5 data is stored in `storage/actions.db` and `storage/notes/`, `storage/summaries/`

### Phase 3 Conversational Fix — Language Encoder

Ada now uses **SentenceTransformers** (MiniLM-L6-v2) for semantic text encoding instead of simple hashing tricks, enabling:

- **Dynamic Confidence Values**: Confidence now varies based on semantic understanding (0.1-0.8 typical range)
- **Better Comprehension**: Real semantic embedding of user input
- **384-Dimensional Embeddings**: Rich vector representations versus basic token counting

#### Expected Behavior Examples:

```
You: hi ada
Ada: Hello, Taahirah! How's your day going? (confidence: 0.47)
You: I'm good  
Ada: That's great to hear. (confidence: 0.59)
You: explain quantum computing
Ada: Quantum computing uses quantum mechanics principles for data processing...(confidence: 0.71)
You: what is the meaning of life
Ada: That's a profound question that has been contemplated throughout human history...(confidence: 0.63)
```

Implementation includes:
- `neural/encoder.py`: Added `LanguageEncoder` class with SentenceTransformers
- `neural/policy_network.py`: Updated default input dimension to 384 
- `core/reasoning.py`: Integrated semantic encoding with fallback compatibility
- `neural/trainer.py`: Updated training to use real language embeddings
- Backward compatibility with existing checkpoints

### Example Log Output

```
🧭 Goal inferred: research whisper.cpp integration (confidence: 0.85)
🚀 Executing plan: plan_research_whisper_cpp_123
🪄 Step 1: Search for information about whisper.cpp integration
✅ Step 1 completed:本地 files found for 'whisper.cpp integration'...
🪄 Step 2: Save research findings to notes
✅ Step 2 completed: Note saved with ID: research_20231027_143022
✅ Success: True
📊 Completed: 2/2 steps
🎯 Reward: +0.82
```
