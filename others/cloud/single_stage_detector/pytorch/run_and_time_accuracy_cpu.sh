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

if [ "x$DATA_DIR" == "x"  ]; then
    echo "DATA_DIR not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x"  ]; then
    echo "MODEL_DIR not set" && exit 1
fi

BATCH_SIZE=1

CONFIG_FILE=""
ARGS=""
if [ "$1" == "int8" ]; then
    ARGS="$ARGS --int8"
    CONFIG_FILE="$CONFIG_FILE --configure $3"
    echo "### running int8 datatype"
elif [ "$1" == "bf16" ]; then
    ARGS="$ARGS --autocast"
    echo "### running bf16 datatype"
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

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
CORES_PER_INSTANCE=$CORES

INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
LAST_INSTANCE=`expr $INSTANCES - 1`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

numa_node_i=0
start_core_i=0
end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`

numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python infer.py --seed 1 --threshold 0.2 -b $BATCH_SIZE -j 0 --data $DATA_DIR --device 0 --checkpoint $MODEL_DIR --no-cuda $ARGS $CONFIG_FILE 2>&1 | tee accuracy_log.txt
accuracy=$(grep 'Accuracy:' ./accuracy_log* |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
echo ""SSD-RN34";"accuracy";$1; ${BATCH_SIZE};${accuracy}" | tee -a ${work_space}/summary.log
