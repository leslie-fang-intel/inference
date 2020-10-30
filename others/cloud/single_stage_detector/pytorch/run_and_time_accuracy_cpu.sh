#!/bin/bash

###############################################################################
### How to run?
### 1) int8 inference accuracy
###    bash run_and_time_accuracy_cpu.sh int8 jit ssd_resnet34.json
### 2) fp32 infenence accuracy
###    bash run_and_time_accuracy_cpu.sh fp32 jit
###
###############################################################################

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export USE_IPEX=1

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

python infer.py --seed 1 --threshold 0.2 --data ${DATASET_DIR} --device 0 --checkpoint $CHECKPOINT --no-cuda --ipex $ARGS $CONFIG_FILE
