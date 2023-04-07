
CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=3746 basicsr/test_prune_l1.py -opt options/test/BasicVSR/test_BasicVSR_Vid4_BIx4.yml --launcher pytorch