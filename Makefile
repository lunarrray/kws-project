# ============================================================================
#  Akylai KWS — project tooling
# ============================================================================
VENV   := venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
HF     := $(VENV)/bin/hf
CONFIG := config.yaml

# --- colors ---
CYAN   := \033[0;36m
GREEN  := \033[0;32m
YELLOW := \033[1;33m
BOLD   := \033[1m
RESET  := \033[0m

.DEFAULT_GOAL := help
.PHONY: help install login dataset clean

help:  ## Show this help
	@printf "$(BOLD)$(CYAN)Akylai KWS — available commands$(RESET)\n\n"
	@printf "  $(GREEN)make install$(RESET)   Create the virtualenv and install pinned dependencies\n"
	@printf "  $(GREEN)make login$(RESET)     Authenticate with the Hugging Face Hub\n"
	@printf "  $(GREEN)make dataset$(RESET)   Extract spectral features (runs utils/feature_extractor.py)\n"
	@printf "  $(GREEN)make clean$(RESET)     Remove caches and build artifacts\n\n"
	@printf "  Config: $(YELLOW)$(CONFIG)$(RESET)\n"

install:  ## Create venv and install dependencies
	@printf "$(BOLD)$(CYAN)==> Creating virtual environment in ./$(VENV)$(RESET)\n"
	@test -d $(VENV) || /opt/homebrew/bin/python3.12 -m venv $(VENV)
	@printf "$(BOLD)$(CYAN)==> Installing pinned requirements$(RESET)\n"
	@$(PIP) install --upgrade pip >/dev/null
	@$(PIP) install -r requirements.txt
	@printf "$(GREEN)✓ Environment ready.$(RESET)\n"

login:  ## Log in to the Hugging Face Hub
	@printf "$(BOLD)$(CYAN)==> Hugging Face Hub login$(RESET)\n"
	@printf "$(YELLOW)Paste a token from https://huggingface.co/settings/tokens$(RESET)\n"
	@$(HF) auth login

dataset:  ## Extract features as configured in config.yaml
	@printf "$(BOLD)$(CYAN)==> Extracting spectral features (config: $(CONFIG))$(RESET)\n"
	@$(PY) -m utils.feature_extractor --config $(CONFIG)
	@printf "$(GREEN)✓ Feature extraction done.$(RESET)\n"

clean:  ## Remove caches and build artifacts
	@printf "$(BOLD)$(YELLOW)==> Cleaning caches$(RESET)\n"
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .ipynb_checkpoints
	@printf "$(GREEN)✓ Clean.$(RESET)\n"
