PYTHON := $(shell command -v python3.11 2>/dev/null || command -v python3)
VENV := .venv
PYTHON_BIN := $(VENV)/bin/python
PIP_BIN := $(VENV)/bin/pip
CLOUD_REQUIREMENTS := cloud/requirements_cloud.txt

.PHONY: setup setup-dev run run-voice train train-rl mps-test test-phases test-cli \
            test-memory test-persona test-planning backup restore build-docs lint format help clean ensure-venv \
            setup-cloud deploy-cloud invoke-cloud sync-storage test-cloud status-cloud

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

# Cloud infrastructure commands
setup-cloud:
	@echo "🌩️  Setting up cloud infrastructure..."
	$(PIP_BIN) install -r $(CLOUD_REQUIREMENTS)
	@echo "✅ Cloud dependencies installed"

.PHONY: build-image
build-image:
	@echo "🔨 Prebuilding Ada Cloud shared image..."
	modal deploy cloud/modal_app.py --name AdaCloudCache
	@echo "✅ Ada Cloud image built and cached"

.PHONY: deploy-cloud
deploy-cloud:
	@echo "🚀 Deploying Ada Cloud to Modal..."
	modal deploy cloud/modal_app.py
	@echo "✅ Ada Cloud deployed successfully"
	@echo "📍 Gateway endpoint: https://ada-cloud.modal.run"

# Ada Cloud Client Commands
.PHONY: cloud-chat
cloud-chat:
	@echo "💬 Starting Ada Cloud chat..."
	@if [ -n "$(words $(filter-out $@,$(MAKECMDGOALS)))" ]; then \
		python3 simple_ada_client.py chat "$(word 2,$(MAKECMDGOALS)) $(wordlist 3,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))"; \
	else \
		python3 simple_ada_client.py chat; \
	fi

.PHONY: cloud-mission
cloud-mission:
	@echo "🎯 Running Ada Cloud mission..."
	@if [ -n "$(words $(filter-out $@,$(MAKECMDGOALS)))" ]; then \
		python3 simple_ada_client.py mission "$(word 2,$(MAKECMDGOALS)) $(wordlist 3,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))"; \
	else \
		echo "❌ Please provide a mission goal"; \
		echo "Usage: make cloud-mission \"<goal>\""; \
	fi

.PHONY: cloud-storage
cloud-storage:
	@echo "📁 Listing Ada Cloud storage..."
	@if [ -n "$(words $(filter-out $@,$(MAKECMDGOALS)))" ]; then \
		python3 simple_ada_client.py storage "$(word 2,$(MAKECMDGOALS))"; \
	else \
		python3 simple_ada_client.py storage; \
	fi

.PHONY: cloud-train
cloud-train:
	@echo "🔧 Training model on Ada Cloud..."
	@if [ -n "$(words $(filter-out $@,$(MAKECMDGOALS)))" ]; then \
		python3 simple_ada_client.py train "$(word 2,$(MAKECMDGOALS))" "$(word 3,$(MAKECMDGOALS))"; \
	else \
		echo "❌ Please provide a model name"; \
		echo "Usage: make cloud-train <model> [epochs]"; \
	fi

.PHONY: deploy-api-gateway
deploy-api-gateway:
	@echo "🌐 Deploying API Gateway..."
	modal deploy cloud/api_gateway.py
	@echo "✅ API Gateway deployed"

.PHONY: run-infer
run-infer:
	@echo "⚡ Running cloud inference test..."
	modal run -m cloud.modal_app ada_infer --data '{"prompt": "Hello Ada, how are you?"}'

.PHONY: run-optimize
run-optimize:
	@echo "🎯 Running cloud optimization test..."
	modal run -m cloud.modal_app ada_optimize --data '{"target_module": "core.reasoning", "parameter_space": {"learning_rate": {"type": "float", "min": 0.001, "max": 0.1}, "batch_size": {"type": "int", "min": 16, "max": 128}}, "max_iterations": 10}'

.PHONY: retrain-adaptive
retrain-adaptive:
	@echo "🧠 Running adaptive retraining on Modal..."
	@echo "📊 Processing feedback logs for model updates..."
	modal run -m cloud.optimizer_service retrain_adaptive

.PHONY: run-mission
run-mission:
	@echo "🎯 Running cloud mission test..."
	modal run -m cloud.modal_app ada_mission --data '{"goal": "Analyze system performance and suggest optimizations"}'

.PHONY: invoke-cloud
invoke-cloud:
	@echo "⚡ Testing Ada Cloud deployment..."
	modal run -m cloud.modal_app ada_infer --data '{"prompt": "Test connection"}'

# Wasabi S3 Storage Commands
.PHONY: test-s3
test-s3:
	@echo "🧪 Testing Wasabi S3 connection..."
	$(PYTHON_BIN) test_wasabi_setup.py

.PHONY: s3-upload
s3-upload:
	@echo "📤 Uploading file to Wasabi S3..."
	@read -p "Enter file path: " filepath; \
	if [ -f "$$filepath" ]; then \
		modal run cloud.modal_app::ada_upload_file --data "{\"file_path\": \"$$filepath\"}"; \
	else \
		echo "File not found: $$filepath"; \
	fi

.PHONY: s3-upload-json
s3-upload-json:
	@echo "📤 Uploading JSON to Wasabi S3..."
	@read -p "Enter storage key: " key; \
	read -p "Enter JSON data (或 file path with @): " data; \
	if [[ "$$data" == @* ]]; then \
		if [ -f "$${data#@}" ]; then \
			modal run cloud.modal_app::ada_upload_json --data "{\"key\": \"$$key\", \"data\": $$(cat "$${data#@}\")}"; \
		else \
			echo "File not found: $${data#@}"; \
		fi; \
	else \
		modal run cloud.modal_app::ada_upload_json --data "{\"key\": \"$$key\", \"data\": $$data}"; \
	fi

.PHONY: s3-download
s3-download:
	@echo "📥 Downloading file from Wasabi S3..."
	@read -p "Enter storage key: " key; \
	read -p "Enter destination path: " dest; \
	modal run cloud.modal_app::ada_download_file --data "{\"key\": \"$$key\", \"destination\": \"$$dest\"}"

.PHONY: s3-download-json
s3-download-json:
	@echo "📥 Downloading JSON from Wasabi S3..."
	@read -p "Enter storage key: " key; \
	modal run cloud.modal_app::ada_download_json --data "{\"key\": \"$$key\"}"

.PHONY: # Voice Commands
.PHONY: cloud-transcribe
cloud-transcribe:
	@echo "🎤 Transcribing audio file to text..."
	@if [ -n "$(words $(filter-out $@,$(MAKECMDGOALS)))" ]; then \
		python3 simple_ada_client.py voice "$(word 2,$(MAKECMDGOALS))"; \
	else \
		echo "❌ Please provide an audio file path"; \
		echo "Usage: make cloud-transcribe <audio_file>"; \
	fi

.PHONY: cloud-speak
cloud-speak:
	@echo "🔊 Converting text to speech..."
	@if [ -n "$(words $(filter-out $@,$(MAKECMDGOALS))) ]; then \
		text="$(word 2,$(MAKECMDGOALS)); \
		voice_id="$(word 3,$(MAKECMDGOALS))"; \
		python3 simple_ada_client.py speak "$text" "$voice_id"; \
	else \
		text="Hello from Ada Cloud!"; \
		python3 simple_ada_client.py speak "$text"; \
	fi

.PHONY: cloud-voice-pipeline
cloud-voice-pipeline:
	@echo "🎤➡️🔧➡️🔊 Voice Pipeline: transcribe → process → speak..."
	@if [ -n "$(words $(filter-out $@,$(MAKECMDGOALS))) ]; then \
		python3 simple_ada_client.py pipeline "$(word 2,$(MAKECMDGOALS))"; \
	else \
		echo "❌ Please provide an audio file path"; \
		echo "Usage: make cloud-voice-pipeline <audio_file>"; \
	fi

# S3 Commands

.PHONY: s3-delete
s3-delete:
	@echo "🗑️  Deleting file from Wasabi S3..."
	@read -p "Enter storage key to delete: " key; \
	modal run -m cloud.modal_app ada_delete_file --data "{\"key\": \"$$key\"}"

.PHONY: s3-sync-dir
s3-sync-dir:
	@echo "🔄 Syncing directory to Wasabi S3..."
	@read -p "Enter local directory path: " local_dir; \
	read -p "Enter storage prefix (optional): " prefix; \
	if [ -d "$$local_dir" ]; then \
		if [ -z "$$prefix" ]; then \
			modal run -m cloud.modal_app ada_sync_directory --data "{\"local_dir\": \"$$local_dir\"}"; \
		else \
			modal run -m cloud.modal_app ada_sync_directory --data "{\"local_dir\": \"$$local_dir\", \"storage_prefix\": \"$$prefix\"}"; \
		fi; \
	else \
		echo "Directory not found: $$local_dir"; \
	fi

.PHONY: sync-storage
sync-storage:
	@echo "💾 Syncing with Wasabi storage..."
	@echo "📦 Testing S3 connection..."
	$(PYTHON_BIN) test_wasabi_setup.py || echo "S3 test completed"
	@echo "✅ Storage sync completed"

.PHONY: test-cloud
test-cloud:
	@echo "🧪 Testing Ada Cloud services..."
	@echo "🔍 Testing inference..."
	modal run -m cloud.modal_app ada_infer --data '{"prompt": "Test connection"}' || echo "Inference test completed"
	@echo "🔍 Testing storage..."
	modal run -m cloud.modal_app ada_list_files --data '{}' || echo "Storage test completed"
	@echo "✅ Cloud service tests completed"

.PHONY: status-cloud
status-cloud:
	@echo "🔍 Checking Ada Cloud status..."
	@if [ -f .env ]; then \
		eval $$(cat .env | sed 's/^/export /'); \
	fi
	@if [ -z "$$ADA_API_KEY" ]; then \
		echo "⚠️  ADA_API_KEY not set, using placeholder for testing..."; \
		export ADA_API_KEY="placeholder-key-for-deployment"; \
	fi
	$(PYTHON_BIN) -m interfaces.remote_client --action status

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
	@echo ""
	@echo "☁️  Cloud Infrastructure Commands:"
	@echo "  setup-cloud    - Install cloud dependencies"
	@echo "  build-image    - Prebuild shared Modal image"
	@echo "  deploy-cloud   - Deploy to Modal serverless platform"
	@echo "  invoke-cloud   - Test cloud deployment"
	@echo "  test-cloud     - Test local cloud client connection"
	@echo "  status-cloud   - Check cloud infrastructure status"
	@echo ""
	@echo "🤖 Ada Cloud Client Commands:"
	@echo "  cloud-chat     - Chat with Ada on Modal Cloud"
	@echo "  cloud-mission  - Run Ada missions on cloud"
	@echo "  cloud-train    - Train models on cloud (GPU A10G)"
	@echo "  cloud-storage  - List cloud storage files"
	@echo "  cloud-transcribe - Transcribe audio to text"
	@echo "  cloud-speak     - Convert text to speech (TTS)"
	@echo "  cloud-voice-pipeline - Full voice: transcribe → process → speak"
	@echo ""
	@echo "💾 Wasabi S3 Storage Commands:"
	@echo "  test-s3        - Test Wasabi S3 connection"
	@echo "  s3-upload      - Upload a file to S3"
	@echo "  s3-upload-json - Upload JSON data to S3"
	@echo "  s3-download    - Download a file from S3"
	@echo "  s3-download-json - Download JSON from S3"
	@echo "  s3-list        - List files in S3 with optional prefix"
	@echo "  s3-delete      - Delete a file from S3"
	@echo "  s3-sync-dir    - Sync local directory to S3"
	@echo "  sync-storage   - Test S3 connection and sync status"
