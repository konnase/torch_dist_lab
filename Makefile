VERSION ?= v0.0.1
HOROVOD_VERSION := horovod0.19.3-tf2.1.0-torch1.4.0-mxnet1.6.0-py3.6-gpu
IMAGE_VERSION := ${HOROVOD_VERSION}-${VERSION}
REGISTRY := registry.sensetime.com/cloudnative4ai/liqingping/torch-dist/cifar

.PHONY: images
images: 
	docker build -t ${REGISTRY}:${IMAGE_VERSION} .

.PHONY: docker-push
docker-push:
	docker push ${REGISTRY}:${IMAGE_VERSION}