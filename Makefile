VENV := .venv
PY := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
PORT ?= 8088
HOST ?= 127.0.0.1

.PHONY: help install install-ce run stop status lint test ci eval eval-latency fetch-jevbench clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "%-16s %s\n", $$1, $$2}'

install: ## 開発用の依存を入れる（Cross-Encoder なし）
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e ".[dev]"

install-ce: ## 推論用の Cross-Encoder（torch / transformers / sentence-transformers）を入れる
	uv pip install --python $(PY) -e ".[ce]"

run: ## ローカルで起動（JE_ALLOW_INSECURE=1 を自分で渡すこと）
	JE_ALLOW_INSECURE=$${JE_ALLOW_INSECURE:-1} $(UVICORN) judgment_engine.app:create_app --factory --host $(HOST) --port $(PORT)

stop: ## 起動中のサーバを止める（listen している PID だけ）
	@pids=$$(lsof -ti tcp:$(PORT) -sTCP:LISTEN); \
	if [ -n "$$pids" ]; then kill $$pids && echo "stopped $$pids"; else echo "not running on $(PORT)"; fi

status: ## 起動確認
	@curl -sS http://$(HOST):$(PORT)/health || echo "not running"

lint: ## ruff
	$(VENV)/bin/ruff check src tests scripts

test: ## テスト
	$(PY) -m pytest

ci: lint test ## CI 相当

eval: ## 評価ハーネス（既定: 同梱の 100 件 + JevBench があればそれも）
	$(PY) -m judgment_engine.eval.cli --authored --out evals/runs/authored-$$(date +%Y%m%d-%H%M%S).json

eval-latency: ## P95 の実測（入力長 × 候補数）
	$(PY) -m judgment_engine.eval.latency --out evals/runs/latency-$$(date +%Y%m%d-%H%M%S).json

fetch-jevbench: ## JevBench（外部データセット）を固定 commit で取得
	$(PY) scripts/fetch_jevbench.py

clean:
	rm -rf var evals/runs
