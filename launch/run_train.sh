#!/bin/bash

BATCH_SIZE=12

# busi
python train.py models/dac/hrnet18_busi_lesion.py --gpus=0 --workers=4 --exp-name=busi --batch-size=$BATCH_SIZE
