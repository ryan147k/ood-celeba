#!/bin/bash
# 使用ex_num控制main函数里调用哪一个实验

ex_num=1
dataset_id=0
CUDA_VISIBLE_DEVICES=6 nohup python -u main.py --debug=False --ex_num=${ex_num} --dataset_id=${dataset_id} > log/ex${ex_num}_${dataset_id}.log 2>&1 &

ex_num=2
dataset_id=0
CUDA_VISIBLE_DEVICES=7 nohup python -u main.py --debug=False --ex_num=${ex_num} --dataset_id=${dataset_id} > log/ex${ex_num}_${dataset_id}.log 2>&1 &

# dataset_id=2
# CUDA_VISIBLE_DEVICES=7 nohup python -u main.py --debug=False --ex_num=${ex_num} --dataset_id=${dataset_id} > log/ex${ex_num}_${dataset_id}.log 2>&1 &
