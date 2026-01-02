IMAGE_NAME ?= preface
REGISTRY ?= quay.io
USERNAME ?= matthdsm
TAG ?= $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' src/preface/__init__.py)
PLATFORMS ?= linux/amd64

.PHONY: build push bump-version test

TEST_OUTDIR_BASE = test_output

test:
	@if [ -z "$(SAMPLESHEET)" ]; then echo "Usage: make test SAMPLESHEET=<path/to/samplesheet.tsv>"; exit 1; fi
	@if [ ! -f "$(SAMPLESHEET)" ]; then echo "Error: samplesheet '$(SAMPLESHEET)' not found."; exit 1; fi
	@rm -rf $(TEST_OUTDIR_BASE)
	@mkdir -p $(TEST_OUTDIR_BASE)

	@echo "Starting PREFACE train permutation tests..."
	@for MODEL_TYPE in neural xgboost svm; do \
		for IMPUTE_TYPE in zero mice mean median knn; do \
			echo "--- Running $$MODEL_TYPE with $$IMPUTE_TYPE (no tune) ---"; \
			pixi run PREFACE train \
				--samplesheet "$(SAMPLESHEET)" \
				--outdir "$(TEST_OUTDIR_BASE)/$${MODEL_TYPE}_$${IMPUTE_TYPE}_no_tune" \
				--model $$MODEL_TYPE \
				--impute $$IMPUTE_TYPE \
				--nsplits 2 \
				--nfeat 10; \
			echo "--- Running $$MODEL_TYPE with $$IMPUTE_TYPE (with tune) ---"; \
			pixi run PREFACE train \
				--samplesheet "$(SAMPLESHEET)" \
				--outdir "$(TEST_OUTDIR_BASE)/$${MODEL_TYPE}_$${IMPUTE_TYPE}_tune" \
				--model $$MODEL_TYPE \
				--impute $$IMPUTE_TYPE \
				--tune \
				--nsplits 2 \
				--nfeat 10; \
		done; \
	done
	@echo "All PREFACE train permutation tests completed."
	
build:
	docker build --platform $(PLATFORMS) -t $(REGISTRY)/$(USERNAME)/$(IMAGE_NAME):$(TAG) .

push:
	docker build --platform $(PLATFORMS) -t $(REGISTRY)/$(USERNAME)/$(IMAGE_NAME):$(TAG) --push .

bump-version:
	@if [ -z "$(v)" ]; then echo "Usage: make bump-version v=1.0.0"; exit 1; fi
	python3 -c "import re; content = open('src/preface/__init__.py').read(); open('src/preface/__init__.py', 'w').write(re.sub(r'__version__ = \".*\"', '__version__ = \"$(v)\"', content))"
	@echo "Version bumped to $(v)"

