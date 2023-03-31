#!/usr/bin/bash
# 
# Copyright (c) 2023-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mkdir -p ./log

ARCH=resnet34
LOSS=sat
DATASET=imagenet
PRETRAIN=0
MOM=0.99
seed=42
epochs=150
batch_size=256


# Parsing arguments
while getopts ":s:a:" flag; do
  case "${flag}" in
    s) seed=${OPTARG};;
    a) ARCH=${OPTARG};;
    :)                                         # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal;;                          # Exit abnormally.
    *)                                         # If unknown (any other) option:
      exit_abnormal;;                          # Exit abnormally.
  esac
done

SAVE_DIR='./log/'

### train
python -u train.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} --manualSeed ${seed}\
       --dataset ${DATASET} --save ${SAVE_DIR}  --epochs ${epochs} --train-batch ${batch_size}\
       --schedule 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 \
       2>&1 | tee -a ${SAVE_DIR}.log

### eval
python -u train.py --arch ${ARCH} --manualSeed ${seed} --sat-momentum ${MOM} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate --epochs ${epochs} --train-batch ${batch_size}\
       --schedule 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 \
       2>&1 | tee -a ${SAVE_DIR}.log