# Ada - Autonomous AI Assistant

Ada is an advanced AI assistant with conversational memory, autonomous planning, self-optimization, and voice interaction capabilities. She can learn from conversations, execute tasks autonomously, and continuously improve her performance.

## Features

### 🧠 Core Capabilities
- **Conversational AI** with semantic understanding
- **Memory System** for contextual awareness
- **Persona Development** that evolves over time
- **Autonomous Planning** for goal execution
- **Tool Integration** for real-world interactions
- **Voice Interface** with speech-to-text and text-to-speech
- **Self-Optimization** through neural evolution

### 🎯 Advanced Features
- **Interactive Reinforcement Learning** - Rate responses to improve Ada
- **Mission Daemon** for continuous background tasks
- **Semantic Memory** with vector-based retrieval
- **Multi-Tool Registry** (file operations, web search, summarization)
- **Web UI Interface** for visual interaction

## Quick Start

### Prerequisites
- Python 3.12+
- macOS (or Linux with minor adjustments)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Engineernoob/Ada.git
cd Ada
```

2. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
make setup
```

### Running Ada

#### CLI Interface
```bash
make run
```

#### Voice Mode (requires additional setup)
```bash
make run-voice
```

#### Web Interface
Open `ui/index.html` in your browser after starting Ada

## Usage Guide

### Interactive Learning
- `/rate good` or `/rate +1` - Reward good responses
- `/rate bad` or `/rate -1` - Penalize poor responses
- View rolling metrics after each rating

### Memory Commands
- `/memory` - Display memory statistics
- Ada automatically remembers conversations with semantic retrieval

### Persona Development
- `/persona` - View current persona characteristics
- Ada develops personality based on conversation patterns

### Autonomous Planning
- `/plan "<goal>"` - Create a plan for a goal
- `/goals` - Show recent autonomous goals
- `/run <plan_id>` - Execute a plan
- `/abort <plan_id>` - Abort a running plan

### Mission Management
- `/mission new "<goal>"` - Create long-running mission
- `/mission list` - View active missions
- `/mission run <mission_id>` - Execute mission
- `/daemon start` - Start background mission daemon
- `/daemon stop` - Stop daemon
- `/daemon status` - Check daemon health
- `/audit` - Force checkpoint audit

### Optimization
- `/optimize now` - Run auto-tuner
- `/evolve` - Launch neural evolution
- `/metrics [n]` - Show latest metrics
- `/rollback <checkpoint_id>` - Restore checkpoint

## Standalone Planner CLI

Plan and execute goals outside the main interface:

```bash
# Create plans
python3 cli_planner.py plan "research topic" --category research
python3 cli_planner.py plan "write article" --execute

# View goals and performance
python3 cli_planner.py goals --limit 10
python3 cli_planner.py performance --insights

# Test tools
python3 cli_planner.py tools --test web_search --input "query"
```

## Voice Mode Setup

For full voice functionality, install additional components:

1. **Whisper.cpp** - Speech-to-text
2. **Piper** - Text-to-speech
3. **Silero VAD** - Voice activity detection

Configure paths in `config/settings.yaml`:

```yaml
voice:
  whisper_path: "/path/to/whisper.cpp"
  piper_path: "/path/to/piper"
  vad_model: "silero_vad.onnx"
  voice_model: "en_US-lessac-medium.onnx"
```

## Web Interface

The `ui/` folder contains a modern web interface:

- Interactive chat interface
- Visual SiriWave animations
- Particle effects background
- Contact management
- Settings configuration
- Real-time response animations

Access the web interface at `ui/index.html` in your browser.

## Training

### Standard Training
```bash
make train
```

### Reinforcement Learning
```bash
make train-rl
```

### Apple Silicon Test
```bash
make mps-test
```

## Architecture

### Core Components
- `core/` - Reasoning engine and settings
- `neural/` - Language models and networks
- `agent/` - Decision-making and execution
- `planner/` - Goal planning and intent handling
- `tools/` - Tool registry and implementations
- `memory/` - Conversation storage and retrieval
- `persona/` - Personality development
- `ui/` - Web interface
- `missions/` - Long-running background tasks
- `optimizer/` - Self-improvement systems
- `storage/` - Data persistence

### Available Tools
- **file_ops** - Read/write/list files
- **web_search** - Information discovery
- **summarize** - Content extraction
- **note** - Journal operations
- **analyze** - Data interpretation

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
# General settings
debug: true
save_interval: 10

# Memory settings
memory:
  max_entries: 10000
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# Persona settings
persona:
  update_interval: 20
  drift_threshold: 0.15

# Mission daemon
daemon:
  sleep_interval: 3600  # 1 hour
  auto_start: false

# Optimizer
optimizer:
  enabled: true
  evolution_generations: 10
  rollback_safe: true

# Voice interface
voice:
  enabled: false
  input_device: "default"
  output_device: "default"
```

## Troubleshooting

### Common Issues

1. **Dependencies**: Ensure all packages are installed with `make setup`
2. **Voice Mode**: Verify audio device permissions on macOS
3. **Memory Issues**: Adjust `max_entries` in settings if using limited RAM
4. **Performance**: Disable optimizer daemon on low-end systems

### Logs

- Main logs: `storage/logs/ada.log`
- Mission logs: `storage/logs/mission.log`
- Optimizer logs: `storage/logs/optimizer.log`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is proprietary software. All rights reserved.

## Support

For issues and questions, please use the GitHub issue tracker.

---

**Note**: Ada is continuously evolving. Check back for updates and new features!
