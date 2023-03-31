#!/usr/bin/bash
# 
# Copyright (c) 2023-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mkdir -p ./log

ARCH=resnet34
LOSS=sat_entropy
DATASET=imagenet100
PRETRAIN=150
MOM=0.9
seed=42
epochs=500
batch_size=64
entropy=0.01


# Parsing arguments
while getopts ":s:e:a:" flag; do
  case "${flag}" in
    s) seed=${OPTARG};;
    e) entropy=${OPTARG};;
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
       --loss ${LOSS} --manualSeed ${seed} --entropy ${entropy}\
       --dataset ${DATASET} --save ${SAVE_DIR}  --epochs ${epochs} --train-batch ${batch_size}\
       2>&1 | tee -a ${SAVE_DIR}.log

### eval
python -u train.py --arch ${ARCH} --manualSeed ${seed}\
       --loss ${LOSS} --dataset ${DATASET} --entropy ${entropy}\
       --save ${SAVE_DIR} --evaluate --epochs ${epochs} --train-batch ${batch_size}\
       2>&1 | tee -a ${SAVE_DIR}.log