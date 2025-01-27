#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=:./$PYTHONPATH

python q_align/evaluate/IQA_dataset_eval.py --model-path compare2score_koniq --device cuda:0
