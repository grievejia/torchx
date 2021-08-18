#!/bin/sh

W0_HOSTNAME=$(echo $VC_SH_0_HOSTS | cut -d'.' -f 1)
echo $W0_HOSTNAME
python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --rdzv_id=12345 --rdzv_backend=c10d --rdzv_endpoint="$W0_HOSTNAME:30001" dist_cifar10/cifar10/trainer.py
