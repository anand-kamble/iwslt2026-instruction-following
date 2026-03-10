.PHONY: install download train eval lint test clean

install:
	pip install -e ".[dev]"

download:
	bash scripts/download_data.sh

train:
	@echo "Usage: make train CONFIG=configs/asr.yaml"
	@test -n "$(CONFIG)" || (echo "ERROR: CONFIG is required" && exit 1)
	python src/train.py --config $(CONFIG)

eval:
	@echo "Usage: make eval CONFIG=configs/asr.yaml CHECKPOINT=path/or/hf-id"
	@test -n "$(CONFIG)" || (echo "ERROR: CONFIG is required" && exit 1)
	@test -n "$(CHECKPOINT)" || (echo "ERROR: CHECKPOINT is required" && exit 1)
	python src/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/ -v

clean:
	rm -rf outputs/ checkpoints/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
