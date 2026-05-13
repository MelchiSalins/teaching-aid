# ==============================================================================
# teaching-aid — developer Makefile (macOS)
# ------------------------------------------------------------------------------
# One-shot setup for a new machine:
#
#     make install     # brew deps + python venv (via uv) + LaTeX packages
#     make smoke       # end-to-end verification (renders a LaTeX scene)
#
# Run `make` (or `make help`) to see every available target.
# ==============================================================================

SHELL        := /bin/bash
.SHELLFLAGS  := -eu -o pipefail -c
.DEFAULT_GOAL := help

# ---- Tooling / paths ---------------------------------------------------------
UV           ?= uv
VENV_DIR     := .venv
PYTHON       := $(VENV_DIR)/bin/python
MANIM        := $(UV) run manim
RUFF         := $(UV) run ruff

# Homebrew packages required to build/run manim on macOS.
BREW_PKGS    := ffmpeg cairo pango pkgconf
BREW_CASKS   := basictex

# LaTeX packages Manim pulls in for MathTex / Matrix / etc.
# Keep in sync with _docs/manim_setup.md.
TLMGR_PKGS   := \
    standalone preview doublestroke relsize fundus-calligra wasysym \
    physics dvisvgm jknapltx rsfs wasy cm-super babel-english gnu-freefont \
    mathastext cbfonts-fd mathalpha everysel ragged2e setspace xcolor

# Colour helpers (only if stdout is a TTY)
ifneq (,$(findstring xterm,$(TERM)))
  C_BOLD := \033[1m
  C_DIM  := \033[2m
  C_OK   := \033[32m
  C_WARN := \033[33m
  C_ERR  := \033[31m
  C_END  := \033[0m
endif

# ==============================================================================
# Help (default target)
# ==============================================================================
.PHONY: help
help: ## Show this help
	@printf "\n$(C_BOLD)teaching-aid — make targets$(C_END)\n\n"
	@awk 'BEGIN {FS = ":.*?## "} \
		/^[a-zA-Z0-9_-]+:.*?## / { printf "  $(C_OK)%-16s$(C_END) %s\n", $$1, $$2 } \
		/^## / { printf "\n$(C_BOLD)%s$(C_END)\n", substr($$0,4) }' $(MAKEFILE_LIST)
	@printf "\n"

# ==============================================================================
## Setup
# ==============================================================================

.PHONY: install setup
install: check-tools brew-deps venv tex-deps ## Full new-machine setup (brew + uv + LaTeX)
	@echo "$(C_OK)✓ Setup complete.$(C_END) Try: make smoke"

setup: install ## Alias for `install`

.PHONY: check-tools
check-tools: ## Verify required CLI tools are installed; print install hints if missing
	@printf "→ Checking required CLI tools…\n"
	@missing=0; \
	check() { \
	  if command -v $$1 >/dev/null 2>&1; then \
	    printf "  $(C_OK)✓$(C_END) %-10s $(C_DIM)(%s)$(C_END)\n" "$$1" "$$(command -v $$1)"; \
	  else \
	    printf "  $(C_ERR)✗$(C_END) %-10s  missing — %s\n" "$$1" "$$2"; \
	    missing=1; \
	  fi; \
	}; \
	check brew "install from https://brew.sh"; \
	check uv   "curl -LsSf https://astral.sh/uv/install.sh | sh"; \
	check make "xcode-select --install"; \
	if [ "$$missing" -ne 0 ]; then \
	  echo "$(C_ERR)Install the tools above, then re-run make.$(C_END)"; \
	  exit 1; \
	fi

.PHONY: brew-deps
brew-deps: ## Install macOS system deps via Homebrew (ffmpeg, cairo, pango, pkgconf, BasicTeX)
	@echo "→ Installing Homebrew formulae: $(BREW_PKGS)"
	brew install $(BREW_PKGS)
	@echo "→ Installing Homebrew casks:    $(BREW_CASKS)"
	@for cask in $(BREW_CASKS); do \
	  if brew list --cask "$$cask" >/dev/null 2>&1; then \
	    echo "  $(C_DIM)already installed: $$cask$(C_END)"; \
	  else \
	    brew install --cask "$$cask"; \
	  fi; \
	done
	@echo "$(C_DIM)  Tip: run  eval \"\$$(/usr/libexec/path_helper)\"  in your shell so \`latex\` is on PATH.$(C_END)"

.PHONY: venv
venv: ## Create .venv and install pinned Python deps via uv
	@echo "→ Syncing Python environment with uv…"
	$(UV) sync
	@echo "$(C_OK)✓ .venv ready at $(VENV_DIR)$(C_END)"

.PHONY: tex-deps
tex-deps: ## Install the LaTeX packages Manim needs (requires sudo — will prompt for password)
	@echo "→ Updating tlmgr itself (sudo required)…"
	sudo tlmgr update --self || true
	@echo "→ Installing LaTeX packages…"
	sudo tlmgr install $(TLMGR_PKGS)

# ==============================================================================
## Code quality
# ==============================================================================

.PHONY: lint
lint: ## Lint Python sources with ruff
	$(RUFF) check .

.PHONY: fmt format
fmt: format
format: ## Format Python sources with ruff
	$(RUFF) format .

# ==============================================================================
## Rendering
# ==============================================================================

.PHONY: smoke
smoke: ## End-to-end smoke test: render the LaTeX scene at 480p
	$(MANIM) -ql latex_smoke.py LatexSmoke
	@echo "$(C_OK)✓ Smoke test passed.$(C_END)"

.PHONY: render-sample
render-sample: ## Render the sample scene (HelloManim) at 480p
	$(MANIM) -ql sample_scene.py HelloManim

# ==============================================================================
## Diagnostics & cleanup
# ==============================================================================

.PHONY: doctor
doctor: ## Print versions of every tool in the stack (for debugging)
	@printf "$(C_BOLD)toolchain versions$(C_END)\n"
	@printf "  uv      : "; $(UV) --version 2>/dev/null              || echo "not installed"
	@printf "  python  : "; $(PYTHON) --version 2>/dev/null          || echo "venv not created — run: make venv"
	@printf "  manim   : "; $(MANIM) --version 2>&1 | head -n1       || echo "manim not installed — run: make venv"
	@printf "  ffmpeg  : "; ffmpeg -version 2>/dev/null | head -n1   || echo "not installed — run: make brew-deps"
	@printf "  latex   : "; latex --version 2>/dev/null | head -n1   || echo "not installed — run: make brew-deps (and open a new shell)"
	@printf "  ruff    : "; $(RUFF) --version 2>/dev/null            || echo "not installed — run: make venv"

.PHONY: clean
clean: ## Remove generated artifacts (media/, caches)
	rm -rf media __pycache__ .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: clean-venv
clean-venv: ## Remove the Python virtualenv (.venv)
	rm -rf $(VENV_DIR)

.PHONY: distclean
distclean: clean clean-venv ## Remove artifacts AND the venv
