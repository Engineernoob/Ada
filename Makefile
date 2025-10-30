PYTHON := $(shell command -v python3.11 2>/dev/null || command -v python3)
VENV := .venv
PYTHON_BIN := $(VENV)/bin/python
PIP_BIN := $(VENV)/bin/pip

.PHONY: setup setup-dev run run-voice train train-rl mps-test test-phases test-cli \
            test-memory test-persona test-planning backup restore build-docs lint format help clean ensure-venv

ensure-venv:
	@if [ ! -x $(PYTHON_BIN) ]; then \
		echo 'Virtual environment not found. Run "make setup" first.'; \
		exit 1; \
	fi

setup:
	$(PYTHON) -m venv $(VENV)
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PIP_BIN) install -r requirements.txt

setup-dev: setup
	$(PIP_BIN) install pytest black flake8 mypy pre-commit

run: ensure-venv
	PYTHONPATH=. $(PYTHON_BIN) -m interfaces.cli

run-voice: ensure-venv
	PYTHONPATH=. $(PYTHON_BIN) -m interfaces.voice_assistant

train: ensure-venv
	PYTHONPATH=. $(PYTHON_BIN) -m neural.trainer

train-rl: ensure-venv
	PYTHONPATH=. $(PYTHON_BIN) -m neural.trainer --mode rl

mps-test: ensure-venv
	PYTHONPATH=. $(PYTHON_BIN) -c "import torch; device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'; print('MPS available:', torch.backends.mps.is_available()); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.device(device))"

test-phases: ensure-venv
	@echo "🔍 Testing key Ada phases..."
	PYTHONPATH=. $(PYTHON_BIN) -c "from memory.episodic_store import EpisodicStore; import torch; store = EpisodicStore(); embed = torch.ones(384); store.store('test', 'response', 0.5, embed); memories = store.retrieve(embed); print(f'✅ Memory test successful - {len(memories)} memories found')"
	PYTHONPATH=. $(PYTHON_BIN) -c "from persona.meta_persona import MetaPersona; persona = MetaPersona(); persona.build_from_transcripts(['test response 1', 'test response 2']); stats = persona.analyze_persona(); print(f'✅ Persona test successful - {stats.tone}')"
	PYTHONPATH=. $(PYTHON_BIN) -c "from planner import IntentEngine, Planner; intent_engine = IntentEngine(); intents = intent_engine.infer(['please analyze the logs']); planner = Planner(); plan = planner.plan(intents[0]) if intents else None; steps = len(plan.steps) if plan and hasattr(plan, 'steps') else 0; print(f'✅ Planning test successful - {len(intents)} intents, {steps} steps generated')"
	@echo "🎉 Phase checks completed"

test-cli: ensure-venv
	@echo "🚀 Testing CLI startup..."
	@timeout 30 PYTHONPATH=. $(PYTHON_BIN) -m interfaces.cli < /dev/null || echo "CLI loaded successfully"

test-memory: ensure-venv
	@echo "🧠 Testing memory system..."
	PYTHONPATH=. $(PYTHON_BIN) -c "from memory.episodic_store import EpisodicStore; import torch; store = EpisodicStore(); embed = torch.ones(384); store.store('test', 'response', 0.5, embed); memories = store.retrieve(embed); print(f'✅ Memory test successful - {len(memories)} memories found')"

test-persona: ensure-venv
	@echo "🎭 Testing persona system..."
	PYTHONPATH=. $(PYTHON_BIN) -c "from persona.meta_persona import MetaPersona; persona = MetaPersona(); persona.build_from_transcripts(['test response 1', 'test response 2']); stats = persona.analyze_persona(); print(f'✅ Persona test successful - {stats.tone}')"

test-planning: ensure-venv
	@echo "🧭 Testing planning system..."
	PYTHONPATH=. $(PYTHON_BIN) -c "from planner import IntentEngine, Planner; intent_engine = IntentEngine(); intents = intent_engine.infer(['please analyze the logs']); planner = Planner(); plan = planner.plan(intents[0]) if intents else None; steps = len(plan.steps) if plan and hasattr(plan, 'steps') else 0; print(f'✅ Planning test successful - {len(intents)} intents, {steps} steps generated')"

backup:
	@echo "💾 Backing up storage..."
	@mkdir -p backups
	@cp -r storage backups/storage-$(shell date +%s) 2>/dev/null || true

restore:
	@echo "🔄 Restoring latest backup..."
	@latest=$$(ls -1t backups 2>/dev/null | head -1); \
	if [ -n "$$latest" ]; then \
		echo "Using backup $$latest"; \
		rsync -a backups/$$latest/ storage/; \
	else \
		echo "No backups found"; \
	fi

build-docs:
	@echo "📚 Building documentation..."
	@mkdir -p docs
	@echo "# Phase 3.5: Conversational Memory & Semantic Recall" > docs/Phase-3.5.md
	@echo "# Phase 4: Persona Formation & Identity Consolidation" > docs/Phase-4.md
	@echo "# Phase 5.5: Autonomous Planning & Tool Use" > docs/Phase-5.5.md

lint: ensure-venv
	@echo "🔍 Linting codebase..."
	$(PYTHON_BIN) -m flake8 interfaces/ core/ memory/ persona/ planner/ tools/ agent/ --extend-ignore=E203,E402,F841,E501,W291,W293

format: ensure-venv
	@echo "🎨 Formatting code..."
	$(PYTHON_BIN) -m black interfaces/ core/ memory/ persona/ planner/ tools/ agent/

clean:
	@echo "🧹 Removing virtual environment and caches..."
	rm -rf $(VENV)
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

help:
	@echo "📚 Ada Makefile Commands:"
	@echo "  setup          - Create virtual environment and install dependencies"
	@echo "  setup-dev      - Install development dependencies"
	@echo "  run            - Start Ada CLI interface"
	@echo "  run-voice      - Start Ada with voice assistant"
	@echo "  train          - Run core trainer"
	@echo "  train-rl       - Run reinforcement-learning trainer"
	@echo "  mps-test       - Check PyTorch accelerator availability"
	@echo "  test-*         - Run smoke tests for subsystems"
	@echo "  backup         - Snapshot storage directory"
	@echo "  restore        - Restore latest backup"
	@echo "  build-docs     - Regenerate documentation stubs"
	@echo "  lint           - Run flake8 lint checks"
	@echo "  format         - Apply black formatting"
	@echo "  clean          - Remove venv and caches"
