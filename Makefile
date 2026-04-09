.PHONY: install run run-local copy-env viz viz-down viz-logs docker-build docker-run clean

COMPOSE        ?= docker compose
COMPOSE_FILES  ?= -f docker-compose.yml -f docker-compose.st.yml
OPEN_BROWSER   ?= 1

# ── Local development ──────────────────────────────────────────────────────────

install:
	uv sync

copy-env:
	@scripts/copy_envexample_to_env.sh

# build/start only the streamlit visualizer (no local uv/python deps)
viz: copy-env
	$(COMPOSE) $(COMPOSE_FILES) up visualizer --build -d --remove-orphans
	@p=$$(grep -E '^[[:space:]]*PORT[[:space:]]*=' .env 2>/dev/null | tail -1 | sed 's/^[^=]*=//' | tr -d '\r'); \
	p=$${p:-8501}; \
	url="http://localhost:$$p/"; \
	echo "streamlit: $$url"; \
	if command -v curl >/dev/null 2>&1; then \
	  i=0; \
	  while [ $$i -lt 60 ]; do \
	    if curl -sf "http://127.0.0.1:$$p/_stcore/health" >/dev/null 2>&1; then break; fi; \
	    i=$$((i+1)); \
	    sleep 0.5; \
	  done; \
	else sleep 2; fi; \
	if [ "$(OPEN_BROWSER)" = "1" ]; then \
	  ( command -v open >/dev/null 2>&1 && open "$$url" ) \
	  || ( command -v xdg-open >/dev/null 2>&1 && xdg-open "$$url" ) \
	  || true; \
	fi

viz-down:
	$(COMPOSE) $(COMPOSE_FILES) stop visualizer

viz-logs:
	$(COMPOSE) $(COMPOSE_FILES) logs -f visualizer

run: viz

run-local:
	PYTHONPATH=src uv run streamlit run src/app/wtp_app.py

# ── Scenario editor (edit adoption curves / emissions trajectories) ────────────

editor:
	PYTHONPATH=src uv run streamlit run src/tools/viz/wip/interface.py

# ── Docker (image without compose) ─────────────────────────────────────────────

IMAGE_NAME ?= globi-use-cases
PORT       ?= 8501

docker-build:
	docker build -f docker/visualizer/Dockerfile -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p $(PORT):8501 $(IMAGE_NAME)

docker-stop:
	docker ps -q --filter ancestor=$(IMAGE_NAME) | xargs -r docker stop

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
