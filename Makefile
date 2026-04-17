.PHONY: install viz viz-down down viz-logs run run-local editor docker-build docker-run docker-stop clean

VIZ_COMPOSE  ?= docker compose -f docker-compose.yml -f docker-compose.st.yml
OPEN_BROWSER ?= 1

install:
	uv sync

.PHONY: copy-env
copy-env: ## Copy `.env.example` to `.env` if `.env` does not exist
	@bash scripts/copy_envexample_to_env.sh

viz: ## run visualizer in docker and open streamlit in a browser
	@make copy-env
	@$(VIZ_COMPOSE) up visualizer --build -d --remove-orphans
	@OPEN_BROWSER=$(OPEN_BROWSER) bash scripts/wait_open_viz.sh

viz-down:
	@$(VIZ_COMPOSE) stop visualizer

down:
	@$(VIZ_COMPOSE) down --remove-orphans

viz-logs:
	@$(VIZ_COMPOSE) logs -f visualizer

run: viz

run-local:
	PYTHONPATH=src uv run streamlit run src/app/wtp_app.py

editor:
	PYTHONPATH=src uv run streamlit run src/tools/viz/wip/interface.py

IMAGE_NAME ?= globi-use-cases
PORT       ?= 8502

docker-build:
	docker build -f docker/visualizer/Dockerfile -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p $(PORT):8502 $(IMAGE_NAME)

docker-stop:
	docker ps -q --filter ancestor=$(IMAGE_NAME) | xargs -r docker stop

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
