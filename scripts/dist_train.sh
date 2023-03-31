#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-4321}

# usage
if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
    exit
fi

python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=2248 \
    basicsr/train_prune_l1.py -opt $CONFIG --launcher pytorch ${@:3}
