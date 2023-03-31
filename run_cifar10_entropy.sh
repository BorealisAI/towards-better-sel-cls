#!/usr/bin/bash
# 
# Copyright (c) 2023-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p ./log

ARCH=vgg16_bn
LOSS=sat_entropy
DATASET=cifar10
PRETRAIN=60
MOM=0.9
seed=42
entropy=0.001


# Parsing arguments
while getopts ":s:e:" flag; do
  case "${flag}" in
    s) seed=${OPTARG};;
    e) entropy=${OPTARG};;
    :)                                         # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal;;                          # Exit abnormally.
    *)                                         # If unknown (any other) option:
      exit_abnormal;;                          # Exit abnormally.
  esac
done

SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_seed-${seed}_entropy-${entropy}

### train
python -u train.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} --manualSeed ${seed} --entropy ${entropy}\
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log

### eval
python -u train.py --arch ${ARCH} --manualSeed ${seed}\
       --loss ${LOSS} --dataset ${DATASET}  --entropy ${entropy}\
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log