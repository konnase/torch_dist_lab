# All-reduce 分布式训练
## 运行方法
### 1. 使用shell脚本简化运行
可以选择直接执行python文件来运行，但是参数多，而且复杂，因此提供了launcher脚本来简化运行步骤。launcher.sh使用方法:
``` shell
# 使用GPU 0 1 2 3 来分布式训练alexnet，可选的模型见下一部分：直接执行python文件手动运行
./launcher.sh -m alexnet -n 4 -g 0,1,2,3 -d cifar100 -f dist
```
### 3. Horovod

``` shell
CUDA_VISIBLE_DEVICES=$Gpus horovodrun -np $Num -H localhost:$Num python3.5 cifar_horovod.py -a $Model --epochs 164 --schedule 81 122 -d $Dataset --gamma 0.1
```

### 2. Torch.distributed

在python指令中，使用CUDA_VISIBLE_DEVICES来分配GPU，注意，分配的GPU个数应该和--nproc_per_node相同
#### AlexNet
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1
```


#### VGG19 (BN)
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1  
```

#### ResNet-110
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 
```

#### ResNet-1202
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a resnet --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 
```

#### PreResNet-110
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a preresnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 
```

#### ResNeXt-29, 8x64d
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a resnext --depth 29 --cardinality 8 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 
```
#### ResNeXt-29, 16x64d
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a resnext --depth 29 --cardinality 16 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 
```

#### WRN-28-10-drop
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a wrn --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 
```

#### DenseNet-BC (L=100, k=12)
**Note**: 
* DenseNet use weight decay value `1e-4`. Larger weight decay (`5e-4`) if harmful for the accuracy (95.46 vs. 94.05) 
* Official batch size is 64. But there is no big difference using batchsize 64 or 128 (95.46 vs 95.11).

```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 
```

#### DenseNet-BC (L=190, k=40) 
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a densenet --depth 190 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 
```

## CIFAR-100

#### AlexNet
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a alexnet --dataset cifar100 --checkpoint checkpoints/cifar100/alexnet --epochs 164 --schedule 81 122 --gamma 0.1 
```

#### VGG19 (BN)
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a vgg19_bn --dataset cifar100 --checkpoint checkpoints/cifar100/vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 
```

#### ResNet-110
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 
```

#### ResNet-1202
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a resnet --dataset cifar100 --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 
```

#### PreResNet-110
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a preresnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 
```

#### ResNeXt-29, 8x64d
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a resnext --dataset cifar100 --depth 29 --cardinality 8 --widen-factor 4  --schedule 150 225 --wd 5e-4 --gamma 0.1 
```
#### ResNeXt-29, 16x64d
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a resnext --dataset cifar100 --depth 29 --cardinality 16 --widen-factor 4  --schedule 150 225 --wd 5e-4 --gamma 0.1 
```

#### WRN-28-10-drop
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a wrn --dataset cifar100 --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 
```

#### DenseNet-BC (L=100, k=12)
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 
```

#### DenseNet-BC (L=190, k=40) 
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3  python3.5 -m torch.distributed.launch --nproc_per_node=2 cifar.py -a densenet --dataset cifar100 --depth 190 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 
```
