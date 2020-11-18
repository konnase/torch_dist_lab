#！/bin/bash

usage() {
    echo "Usage:"
    echo "  launcher.sh [-m Model] [-n Number of Gpus] [-g Gpus] [-d Dataset] [-f framework(horovod, dist)]"
    exit -1
}

while getopts 'm:n:g:d:f:' OPT; do
    case $OPT in
    m) Model=$OPTARG ;;
    g) Gpus=$OPTARG ;;
    n) Num=$OPTARG ;;
    d) Dataset=$OPTARG ;;
    f) Framework=$OPTARG ;;
    ?) usage ;;
    esac
done

echo model: $Model
echo num of gpus: $Num
echo gpus: $Gpus
echo dataset: $Dataset
echo framework: $Framework

horovod="horovod"
dist="dist"

if [ "$Framework" == "$horovod" ]; then
    CUDA_VISIBLE_DEVICES=$Gpus horovodrun -np $Num -H localhost:$Num python cifar_horovod.py -a $Model --epochs 164 --schedule 81 122 -d $Dataset --gamma 0.1
elif [ "$Framework" == "$dist" ]; then
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$Gpus python -m torch.distributed.launch --nproc_per_node=$Num cifar.py -a $Model --epochs 164 --schedule 81 122 -d $Dataset --gamma 0.1
fi
