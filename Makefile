IMAGE_NAME ?= preface
REGISTRY ?= quay.io
USERNAME ?= matthdsm
TAG ?= $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' src/preface/__init__.py)

.PHONY: build push

build:
	docker build -t $(REGISTRY)/$(USERNAME)/$(IMAGE_NAME):$(TAG) .

push:
	docker push $(REGISTRY)/$(USERNAME)/$(IMAGE_NAME):$(TAG)
