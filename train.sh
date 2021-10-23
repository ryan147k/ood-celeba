#!/bin/bash
# 使用ex_num控制main函数里调用哪一个实验

#ex_num=1
#tcl=(1 2 3)
#cuda=(0 1 3)
#for i in  0 1 2
#do
#  batch_size=256
#  dataset_type=0
#  train_class=${tcl[${i}]}
#  CUDA_VISIBLE_DEVICES=${cuda[${i}]} nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_type=${dataset_type} --train_class=${train_class} > log/ex${ex_num}_${dataset_type}_${train_class}.log 2>&1 &
#done


#ex_num=2
#tcl=(1 2 3)
#cuda=(0 1 4)
#for i in 0 1 2
#do
#  batch_size=512
#  dataset_type=0
#  train_class=${tcl[${i}]}
#  CUDA_VISIBLE_DEVICES=${cuda[${i}]} nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_type=${dataset_type} --train_class=${train_class} > log/ex${ex_num}_${dataset_type}_${train_class}.log 2>&1 &
#done

ex_num=3
tcl=(1 2 3)
cuda=(0 1 4)
for i in 0 1 2
do
  batch_size=128
  dataset_type=0
  train_class=${tcl[${i}]}
  CUDA_VISIBLE_DEVICES=${cuda[${i}]} nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_type=${dataset_type} --train_class=${train_class} > log/ex${ex_num}_${dataset_type}_${train_class}.log 2>&1 &
done