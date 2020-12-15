# FROM horovod/horovod:0.19.3-tf2.1.0-torch-mxnet1.6.0-py3.6-gpu
FROM registry.sensetime.com/cloudnative4ai/baseimages/pytorch:1.5.0-cuda9.2-devel-cudnn7.6-nccl2.4.8-py3.6-centos7.8
LABEL maintainer="liqinping@sensetime.com" 

RUN mkdir -p /torch-dist/cifar && mkdir -p /torch-dist/cifar/cifar
WORKDIR /torch-dist/cifar

ADD models models
ADD utils utils
ADD cifar.py cifar.py
ADD cifar_horovod.py cifar_horovod.py
ADD cifar_multi_nodes.py cifar_multi_nodes.py
ADD launcher.sh launcher.sh
ADD cifar/cifar-10-python.tar.gz cifar/

# docker build -t registry.sensetime.com/cloudnative4ai/pytorch/cifar-ddp:latest .