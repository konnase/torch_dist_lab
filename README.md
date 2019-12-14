# 使用DistributedDataParallel(DDP)和Horovod两种ring-allreduce分布式训练框架(单机多卡)

## ring-allreduce简介

#### 数据并行

数据并行的方法包含在多个节点上并行分割数据和训练。不同批次的数据在不同节点上分别被计算，反向传播得到梯度，然后节点之间通过通信，求得平均梯度，用来对每个节点中的模型副本进行一致化更新。

具体的步骤可以简化为以下几步：

+ 运行训练模型的多个副本，每个副本：
  + 读取数据块
  + 将其输入模型
  + 计算梯度
+ 通过通信，计算所有副本的梯度均值
+ 一致化更新所有模型副本
+ 重复上述步骤



#### ring-allreduce

数据并行这种分布式训练的关键是找到一个好的通信策略。ring-allreduce是一个稳定的通信策略，GPU被组织成了一个逻辑环，每个GPU只有一个左邻和一个右邻；每个GPU只会向它的右邻居发送数据，并从他的左邻居接收数据。

<img src="./doc/fig1.png" alt="image" style="zoom:40%;" />

ring-allreduce分为两个步骤，分别是**The Scatter-Reduce**和**The Allgather**。

+ The Scatter-Reduce

  假设有4块GPU，梯度数据将被分为4块，GPU将进行4-1次Scatter-Reduce迭代，每次迭代中，GPU向右邻居发送一个块，并从左邻居接收一个块。第n个GPU从发送块n和接收块n-1开始，每次迭代都发送它在前一次迭代中接收到的块。

  ![image](./doc/fig2.png)

  

+ The Allgather

  在The scatter-reduce结束后，每个GPU都有一块数据是最终值，接着，GPU交换这些块，以便所有的GPU都具有所需的数据，交换的过程和Scatter-reduce类似

  ![image](./doc/fig3.png)



之后，每个GPU对梯度求平均，并更新模型，读取下一批数据，进行训练。

## 两种框架的使用

### DDP

Pytorch的torch.distributed模块中封装了ring-allreduce算法，是官方推荐的分布式训练方法。使用它的具体模式如下：

``` python
import torch.distributed as dist

# Use CUDA
use_cuda = torch.cuda.is_available()
# 当前线程所处的rank
local_rank = dist.get_rank()
# 设定cuda的默认GPU，每个rank不同
torch.cuda.set_device(local_rank) 
# 初始化分布式进程组，backend指定后端通信方式，包括mpi，gloo，nccl。
# init_method是一个url，指定如何初始化进程组(如何找到其他节点)
torch.distributed.init_process_group(backend='nccl',init_method="env://")

def main():
    # 训练集
    trainset = ...
    # 分布式采样器
    sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # 加载训练集
    trainloader = data.DataLoader(dataset=trainset, batch_size=args.train_batch, shuffle=False, sampler=sampler)
		# 测试集
    testset = ...
    # 在分布式训练中，一个batch会被划分为几等份，分配给每个GPU进行训练。
    # 因此，batch_size需要乘上GPU的个数
    testloader = data.DataLoader(testset, batch_size=args.test_batch * dist.get_world_size(), shuffle=False, num_workers=args.workers)
    # 准备模型
    model = ...
    # 将模型放在自己的rank对应的cuda上
    device = torch.device('cuda', local_rank)
    model = model.to(device)
    # 并行化
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)    
    # 损失函数和优化方法
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        train...
				eval...

if __name__ == '__main__':
    main()

```

训练时，需要在指令中分配GPU，确定进程个数等，训练指令可以是：

``` shell
CUDA_VISIBLE_DEVICES=0,1,2,3  python3.5 -m torch.distributed.launch --nproc_per_node=4 train.py [other arguments]
```

### horovod

horovod是Uber开源的使用ring-allreduce算法分布式训练框架，适用于多个机器学习框架。

Pytorch中使用horovod的模式如下：

``` python
import horovod.torch as hvd
# 初始化
hvd.init()
# 分配rank
local_rank = hvd.local_rank()
torch.cuda.set_device(local_rank)

def main():
    # 训练集
    trainset = ...
    # 分布式采样器
    sampler = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=hvd.size(), rank=hvd.rank())
    # 加载训练集
    trainloader = data.DataLoader(dataset=trainset, batch_size=args.train_batch, shuffle=False, sampler=sampler)
		# 测试集
    testset = ...
    testloader = data.DataLoader(testset, batch_size=args.test_batch * hvd.size(), shuffle=False, num_workers=args.workers)
    # 准备模型
		model=...
    # 将模型放到指定cuda
    device = torch.device('cuda', local_rank)
    model = model.to(device)
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 用horovod封装优化器
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # 广播参数，这个是为了在一开始的时候同步各个gpu之间的参数
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        train...
        eval...
        
if __name__ == '__main__':
    main()

```



训练时也需要在指令中分配GPU，确定进程个数

``` shell
CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 -H localhost:4 python3.5 train.py [other arguments]
```



## 效果对比

![image](./doc/fig4.png)

