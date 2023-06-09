#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-4321}

# usage
if [ $# -ne 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_test.sh [number of gpu] [path to option file]"
    exit
fi

#PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=3459 \
    basicsr/test_prune_l1.py -opt $CONFIG --launcher pytorch
