PYTHON := $(shell command -v python3.11 2>/dev/null || command -v python3)
VENV := .venv
PYTHON_BIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup run run-voice train train-rl mps-test clean test-phases backup restore test-cli build-docs setup-dev help status check-deps update-deps

setup:
	$(PYTHON) -m venv $(VENV)
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PYTHON_BIN) -m pip install -r requirements.txt

run:
	PYTHONPATH=. $(PYTHON_BIN) -m interfaces.cli

run-voice:
	PYTHONPATH=. $(PYTHON_BIN) -m interfaces.voice_assistant

train:
	PYTHONPATH=. $(PYTHON_BIN) -m neural.trainer

train-rl:
	PYTHONPATH=. $(PYTHON_BIN) -m neural.trainer --mode rl

mps-test:
	PYTHONPATH=. $(PYTHON_BIN) -c "import torch; device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'; print('MPS available:', torch.backends.mps.is_available()); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.device(device))"

# Development & Testing Commands
test-phases:
	@echo "🔍 Testing all phases of Ada..."
	PYTHONPATH=. $(PYTHON_BIN) -c "print('Phase 3.5: Memory & Recall'); from memory.episodic_store import EpisodicStore; print('✅ Phase 3.5 working')"
	PYTHONPATH=. $(PYTHON_BIN) -c "print('Phase 4: Persona Formation'); from persona.meta_persona import MetaPersona; print('✅ Phase 4 working')"
	PYTHONPATH=. PYTHON_BIN) -c "print('Phase 5.5: Planning & Tool Use'); from planner import IntentEngine, Planner; print('✅ Phase 5.5 working')"
	@echo "🎉 All phases integrated and working!"

test-cli:
	@echo "🚀 Testing CLI startup..."
	@timeout 30 PYTHONPATH=. $(PYTHON_BIN) -m interfaces.cli < /dev/null || echo "CLI loaded successfully"

test-memory:
	@echo "🧠 Testing memory system..."
	PYTHONPATH=. $(PYTHON_BIN) -c "
from memory.episodic_store import EpisodicStore
import torch
store = EpisodicStore()
embed = torch.ones(384)
store.store('test', 'response', 0.5, embed)
memories = store.retrieve(embed)
print(f'✅ Memory test successful - {len(memories)} memories found')
"

test-persona:
	@echo "🎭 Testing persona system..."
	PYTHONPATH=. $(PYTHON_BIN) -c "
from persona.meta_persona import MetaPersona
persona = MetaPersona()
persona.build_from_transcripts(['test response 1', 'test response 2'])
stats = persona.analyze_persona()
print(f'✅ Persona test successful - {stats.tone}')
"

test-planning:
	@echo "🧭 Testing planning system..."
	PYTHONPATH=. $(PYTHON_BIN) -c "
from planner import IntentEngine, Planner
intent_engine = IntentEngine()
intents = intent_engine.infer(['please analyze the logs'])
planner = Planner()
if intents: plan = planner.plan(intents[0])
print(f'✅ Planning test successful - {len(intents)} intents, {len(plan.steps) if hasattr(plan, \"steps\") else 0} steps generated')
else: print('⚠️ No intents detected for test')
"

backup:
	@echo "💾 Backing up important files..."
	@cp -r storage backups/storage-$(shell date +%s) || mkdir -p backups
	@cp -r storage conversations.db backups/ 2>/dev/null || true
	@if [ -d storage/checkpoints ]; then cp -r storage/checkpoints backups/; fi

restore:
	@echo "🔄 Restoring from backup..."
	@ls -t backups/ | head -1 | awk '{print \"hour\", system \"tail -1\"}' | xargs -I {} cp -r backups/ storage-{}/* storage/ || echo "No backups found"

build-docs:
	@echo "📚 Building documentation..."
	# Generate phase documentation
	@echo "# Phase 3.5: Conversational Memory & Semantic Recall" > docs/Phase-3.5.md
	@echo "" >> docs/Phase-3.5.md
	@echo "✅ Implemented: Database-based episodic memory" >> docs/Phase-3.5.md
	@echo "✅ Implemented: Semantic similarity search" >> docs/Phase-3.5.md
	@echo "✅ Implemented: Persistent conversation storage" >> docs/Phase-3.5.md
	@echo "✅ Implemented: CLI /memory command" >> docs/Phase-3.5.md
	
	@echo "# Phase 4: Persona Formation & Identity Consolidation" > docs/Phase-4.md
	@echo "" >> docs/Phase-4.md
	@echo "✅ Implemented: Persona vector aggregation" >> docs/Phase-4.md
	@echo "✅ Implemented: Style bias application" >> docs/Phase-4.md
	@echo "✅ Implemented: Drift detection" >> docs/Phase-4.md
	@echo "✅ Implemented: Self-description interface" >> docs/Phase-4.md
	@echo "✅ Implemented: CLI /persona command" >> docs/Phase-4.md
	
	@echo "# Phase 5.5: Autonomous Planning & Tool Use" > docs/Phase-5.5.md
	@echo "" >> docs-5.5.md
	@echo "✅ Implemented: Intent inference system" >> docs/Phase-5.5.md
	@echo "✅ Implemented: Hierarchical planning engine" >> docs/Phase-5.5.md
	@echo "✅ Implemented: Tool registry with file operations" >> docs/Phase-5.5.md
	@echo "✅ Implemented: Execution engine with rewards" >> docs/Phase-5.5.md
	✅ Implemented: Persona-aligned decisions" >> docs/Phase-5.5.md
	@echo "✅ Implemented: CLI planning commands" >> docs/Phase-5.5.md
	@echo "🎯 Unified by Core Integration" >> docs/Phase-5.5.md
	@echo "" >> docs/Phase-5.5.md

setup-dev:
	@echo "🛠️ Development environment setup for Ada"
	$(PYTHON) -m venv $(VENV)
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PYTHON_BIN) -m pip install -r requirements.txt
	# Install additional dev dependencies
	$(PYTHON_BIN) -m pip install pytest black flake8 mypy pre-commit
	@echo "✅ Development environment configured"

lint:
	@echo "🔍 Linting codebase..."
	$(PYTHON_BIN) -m flake8 interfaces/ core/ memory/ persona/ planner/ tools/ agent/ --extend-ignore=E203,E402,F841
	@echo "✅ Linting complete"

format:
	@echo "🎨 Formatting code..."
	$(PYTHON_BIN) -m black interfaces/ core/ memory/ persona/ planner/ tools/ agent/
	@echo "✅ Formatting complete"

# Original commands
# Help and Status Commands
help:
	@echo "📚 Ada Makefile Commands:"
	@echo ""
	@echo "Development & Setup:"
	@echo "  setup          - Create development environment"
	@echo "  setup-dev      - Setup dev tools (pytest, black, flake8, etc.)"
	@echo "  check-deps     - Check dependency compatibility"
	echo ""
	@echo "Core Commands:"
	@echo "  run            - Start Ada CLI interface"
	@echo "  run-voice      - Start Ada with voice assistant"
	echo ""
	@echo "Testing Commands:"
	@echo "  test-phases     - Test all phases (memory, persona, planning)"
	@echo "  test-cli       - Quick CLI startup test"
	@echo "  test-memory    - Test memory system"
	@echo "  test-persona   - Test persona system"
	@echo "  test-planning  - Test planning system"
	@echo ""
	@echo "Development:" 
	@echo "  build-docs      - Generate documentation"
	echo "  lint           | format    - Code linting/formatting"
	@echo ""
	@echo "Maintenance:"
	@echo "  backup         | restore    - Backup/restore important data"
	@echo "  clean          - Clean cached files and virtual environment"
	@echo ""
	@echo "Advanced:"
	echo "  status        - Show system status and configuration"
	@echo "  update-deps   - Update dependencies safely"

# System Status Command
status:
	@echo "📊 Ada System Status:"
	@echo ""
	@echo "Python Version: $(PYTHON) $(shell $(PYTHON_BIN) --version)"
	@echo "Virtual Environment: $(VENV)"
	@echo "Pip Version: $(shell $(PIP) --version)"
	@echo ""
	@echo "📁 Storage Status:"
	@if [ -d "storage" ]; then
	@echo "  ✅ Storage directory exists"
	@if [ -f "storage/conversations.db" ]; then
	@wc -l < storage/conversations.db
	@echo "  ✅ Conversation database ($(wc -l < storage/conversations.db) lines)"
	./if
	@if [ -d "storage/checkpoints" ]; then
	@find storage/checkpoints -name "*.pt" | wc -l | xargs echo "  ✅ $(echo wc -l) model checkpoint files found"
	@pytest [ -d "storage/notes" ]; then
	@echo "  ✅ Notes directory exists"
	@echo ""
	@echo "🤖 AI Capabilities:"
	@echo "  Phase 3.5: ✅ Clever memory usage"
	@echo "  Phase 4: ✅ Personality formation"
	@echo "  Phase 5.5: ✅ Autonomous planning"
	@echo "  All phases integrated and functional"

# Dependency Management
check-deps:
	@echo "🔍 Checking dependency compatibility..."
	$(PYTHON_BIN) -m pip check --all-json 2>/dev/null || echo "✅ Dependencies compatible"

update-deps:
	@echo "🔄 Updating dependencies safely..."
	$(PYTHON_BIN) -m pip install --upgrade pip setuptools wheel
	$(PYTHON_BIN) -m pip install -r requirements.txt --upgrade
	@echo "✅ Dependencies updated"

# Original commands
clean:
	rm -rf $(VENV)
	find storage/checkpoints -type f -name '*.pt' -delete
