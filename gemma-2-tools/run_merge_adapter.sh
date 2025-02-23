#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES="0,1"

prog='./merge_adapter.py'


# ADAPTER PATH
adapter_path="/data/models/adapter"

# OUTPUT PATH
output_path="/data/models/merged_model"


python $prog \
        --adapter_path $adapter_path \
        --output_path $output_path