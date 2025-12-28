IMAGE_NAME ?= preface
REGISTRY ?= quay.io
USERNAME ?= matthdsm
TAG ?= $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' src/preface/__init__.py)
PLATFORMS ?= linux/amd64

.PHONY: build push bump-version
	
build:
	docker build --platform $(PLATFORMS) -t $(REGISTRY)/$(USERNAME)/$(IMAGE_NAME):$(TAG) .

push:
	docker build --platform $(PLATFORMS) -t $(REGISTRY)/$(USERNAME)/$(IMAGE_NAME):$(TAG) --push .

bump-version:
	@if [ -z "$(v)" ]; then echo "Usage: make bump-version v=1.0.0"; exit 1; fi
	python3 -c "import re; content = open('src/preface/__init__.py').read(); open('src/preface/__init__.py', 'w').write(re.sub(r'__version__ = \".*\"', '__version__ = \"$(v)\"', content))"
	@echo "Version bumped to $(v)"

