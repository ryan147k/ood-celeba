#!/bin/bash
# 使用ex_num控制main函数里调用哪一个实验

ex_num=4
dataset_id=0
batch_size=256
CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_id=${dataset_id} > log/ex${ex_num}_${dataset_id}.log 2>&1 &

#ex_num=5
#dataset_id=1
#batch_size=256
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_id=${dataset_id} > log/ex${ex_num}_${dataset_id}.log 2>&1 &
#
#ex_num=5
#dataset_id=2
#batch_size=256
#CUDA_VISIBLE_DEVICES=7 nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_id=${dataset_id} > log/ex${ex_num}_${dataset_id}.log 2>&1 &
