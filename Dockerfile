FROM horovod/horovod:0.19.3-tf2.1.0-torch-mxnet1.6.0-py3.6-gpu
LABEL maintainer="liqinping@sensetime.com" 

RUN mkdir -p /torch-dist/cifar
WORKDIR /torch-dist/cifar

ADD models models
ADD utils utils
ADD cifar.py cifar.py
ADD cifar_horovod.py cifar_horovod.py
ADD cifar_multi_nodes.py cifar_multi_nodes.py
ADD launcher.sh launcher.sh
ADD cifar/cifar-10-python.tar.gz cifar/cifar-10-python.tar.gz
ADD cifar/cifar-100-python.tar.gz cifar/cifar-100-python.tar.gz
