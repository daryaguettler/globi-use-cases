.PHONY: install run run-local copy-env docker-build docker-run clean

# ── Local development ──────────────────────────────────────────────────────────

install:
	uv sync

# optional hook before compose (e.g. cp .env.example .env) — extend as needed
copy-env:
	@true

# run in docker: no local python/uv deps; image installs deps via uv in the build
run: copy-env
	PORT=$(PORT) docker compose up app --build

run-local:
	PYTHONPATH=src uv run streamlit run src/app/wtp_app.py

# ── Scenario editor (edit adoption curves / emissions trajectories) ────────────

editor:
	PYTHONPATH=src uv run streamlit run src/tools/viz/wip/interface.py

# ── Docker ─────────────────────────────────────────────────────────────────────

IMAGE_NAME ?= globi-use-cases
PORT       ?= 8501

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p $(PORT):8501 $(IMAGE_NAME)

docker-stop:
	docker ps -q --filter ancestor=$(IMAGE_NAME) | xargs -r docker stop

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
