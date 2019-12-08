#！/bin/bash

usage() {
    echo "Usage:"
    echo "  launcher.sh [-m Model] [-n Number of Gpus] [-g Gpus] -[d Dataset]"
    exit -1
}

while getopts 'm:n:g:d:' OPT; do
    case $OPT in
        m) Model=$OPTARG;;
        g) Gpus=$OPTARG;;
        n) Num=$OPTARG;;
        d) Dataset=$OPTARG;;
        ?) usage;;
    esac
done

echo model: $Model
echo num of gpus: $Num
echo gpus: $Gpus
echo dataset: $Dataset

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$Gpus  python3.5 -m torch.distributed.launch --nproc_per_node=$Num cifar.py -a $Model --epochs 164 --schedule 81 122 -d $Dataset --gamma 0.1
