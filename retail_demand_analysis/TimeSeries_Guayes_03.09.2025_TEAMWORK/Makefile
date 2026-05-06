# ------------------------------------------------------------
# Favorita Forecasting - Developer Makefile
# Authors: Agus • Kiko • Patrick
# ------------------------------------------------------------
# ------------------------------------------------------------
# Python version enforcement
# ------------------------------------------------------------

PYTHON_VERSION := 3.12
PYTHON_BIN := $(shell which python3.12 || which python3)


hello:
	@echo "Hello from Agus, Kiko and Patrick! 🚀"

# Detect OS
UNAME_S := $(shell uname 2>/dev/null)

ifeq ($(OS),Windows_NT)
    PYTHON := python
    ACTIVATE := .venv\Scripts\activate
else
    PYTHON := python3
    ACTIVATE := source .venv/bin/activate
endif


.PHONY: help setup install-dev hooks setup-dev clean

help:
	@echo ""
	@echo "================ FAVORITA DEV MAKEFILE ================"
	@echo ""
	@echo "=========== Authors: Agus - Kiko - Patrick ============"
	@echo ""
	@echo " Available commands:"
	@echo ""
	@echo "  make setup         Create virtual environment (.venv)"
	@echo "  make install       Install project with dev deps"
	@echo "  make hooks         Activate git pre-commit hooks"
	@echo "  make setup-dev     Full setup (venv + install + hooks)"
	@echo "  make clean         Remove virtual environment"
	@echo ""
	@echo " After setup, activate manually:"
ifeq ($(OS),Windows_NT)
	@echo "   .venv\\Scripts\\activate"
else
	@echo "   source .venv/bin/activate"
endif
	@echo "========================================================"
	@echo ""



setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created."
	@echo "Activate with:"
	@echo "  $(ACTIVATE)"
	$(ACTIVATE)


install:
	@echo "Installing development dependencies..."
ifeq ($(OS),Windows_NT)
	pip install -e .[dev]
else
	pip install -e ".[dev]"
endif
	@echo "Done."


hooks:
	@echo "Installing git pre-commit hooks..."
	pre-commit install
	@echo "Hooks installed."


setup-dev: setup install hooks
	@echo "Development environment ready 🎯"


clean:
	@echo "Removing virtual environment..."
ifeq ($(OS),Windows_NT)
	rmdir /S /Q .venv
else
	rm -rf .venv
endif
	@echo "Done."


venv12:
	@echo "Creating virtual environment with Python $(PYTHON_VERSION)..."
	@$(PYTHON_BIN) -m venv .venv
	@echo "Virtual environment created."

.PHONY: active
active:
	@bash -lc "source .venv/bin/activate && exec bash"

run app:
	streamlit run src/streamlit_app/app.py