#!/bin/bash

###############################################################################
### How to run?
### option step(for int8 calibration): bash run_and_time_accuracy_cpu.sh int8 jit ssd_resnet34.json calibration 3
### 1) int8 inference accuracy
###    bash run_and_time_accuracy_cpu.sh int8 jit ssd_resnet34.json
### 2) fp32 infenence accuracy
###    bash run_and_time_accuracy_cpu.sh fp32 jit
###
###############################################################################

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export USE_IPEX=1

BATCH_SIZE=32

DATASET_DIR='/lustre/dataset/COCO2017'
CHECKPOINT='../pretrained/resnet34-ssd1200.pth'

CONFIG_FILE=""
ARGS=""
if [ "$1" == "int8" ]; then
    ARGS="$ARGS --int8"
    CONFIG_FILE="$CONFIG_FILE --configure-dir $3"
    echo "### running int8 datatype"
else
    echo "### running fp32 datatype"
fi

if [ "$2" == "jit" ]; then
    ARGS="$ARGS --jit"
    echo "### running jit fusion path"
else
    echo "### running native path"
fi

if [ "$4" == "calibration"  ]; then
    ARGS="$ARGS --calibration"
    echo "### running int8 calibration"
fi

if [ -n "$5" ]; then
    ARGS="$ARGS --iter $5"
fi

python infer.py --seed 1 --threshold 0.2 -b $BATCH_SIZE -j 0 --data ${DATASET_DIR} --device 0 --checkpoint $CHECKPOINT --no-cuda --ipex $ARGS $CONFIG_FILE
