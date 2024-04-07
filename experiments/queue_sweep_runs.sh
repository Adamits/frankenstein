#!/bin/bash

K=$1
DATASET=$2
LANGUAGE=$3
ARCH=$4
SWEEP_ID=$5
WANDB_PROJECT=$6

for i in $(seq 1 $K); do
    sbatch --export=ALL,DATASET=${DATASET},LANGUAGE=${LANGUAGE},ARCH=${ARCH},SWEEP_ID=${SWEEP_ID},WANDB_PROJECT=${WANDB_PROJECT} experiments/train_sweep_agent.sh;
done