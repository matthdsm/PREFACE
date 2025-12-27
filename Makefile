IMAGE_NAME ?= preface
REGISTRY ?= quay.io
USERNAME ?= matthdsm
TAG ?= latest

.PHONY: build push

build:
	docker build -t $(REGISTRY)/$(USERNAME)/$(IMAGE_NAME):$(TAG) .

push:
	docker push $(REGISTRY)/$(USERNAME)/$(IMAGE_NAME):$(TAG)
