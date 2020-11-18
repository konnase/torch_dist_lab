FROM registry.sensetime.com/cloudnative4ai/horovod/horovod:0.19.3-tf2.1.0-torch-mxnet1.6.0-py3.6-gpu
LABEL maintainer="liqinping@sensetime.com" 

RUN mkdir -p /torch-dist/cifar
WORKDIR /torch-dist/cifar
ADD . .

