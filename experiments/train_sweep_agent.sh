#!/bin/bash

#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=20gb
#SBATCH --time=24:00:00
##SBATCH --qos=blanca-kann
#SBATCH --qos=preemptable
#SBATCH --constraint=V100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/run_sweep_agent.%j.log

source /curc/sw/anaconda3/latest
conda activate hyperparameter-sensitivity

# SET AS ENV VARS FROM CALLING SCRIPT
# DATASET
# LANGUAGE
# ARCH
# SWEEP_ID
# WANDB_PROJECT

# Fixes path issue.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/adwi9965/software/anaconda/envs/yoyodyne/lib/

readonly ROOT="/projects/adwi9965/frankenstein"
readonly DATA="${ROOT}/data/${DATASET}"

if [[ "${DATASET}" == "g2p" ]]
then
    # We only do the medium setting for G2P (high is English-only and very large data)
    readonly TRAIN="${DATA}/${LANGUAGE}.train"
    readonly DEV="${DATA}/${LANGUAGE}.dev"
    readonly FEATURES_COL=0
    readonly TARGET_COL=2
elif [[ "${DATASET}" == "sigmorphon-2017-inflection" ]]
then
    # We only do high for inflection
    readonly TRAIN="${DATA}/${LANGUAGE}-train-high"
    readonly DEV="${DATA}/${LANGUAGE}-dev"
    readonly FEATURES_COL=3
    readonly TARGET_COL=2
elif [[ "${DATASET}" == "sigmorphon-2023-inflection" ]]
then
    readonly TRAIN="${DATA}/${LANGUAGE}.trn"
    readonly DEV="${DATA}/${LANGUAGE}.dev"
    readonly FEATURES_COL=2
    readonly TARGET_COL=3
elif [[ "${DATASET}" == "bollman-2019-histnorm" ]]
then
    readonly TRAIN="${DATA}/${LANGUAGE}.train.txt"
    readonly DEV="${DATA}/${LANGUAGE}.dev.txt"
    readonly FEATURES_COL=0
    readonly TARGET_COL=2
elif [[ "${DATASET}" == "transliteration" ]]
then
    echo transliteration not yet implemented. We need to implement new eval methods for dev.
    exit 1;
    readonly TRAIN="${DATA}/${LANGUAGE}.train.tsv"
    readonly DEV="${DATA}/${LANGUAGE}.dev.tsv"
    readonly FEATURES_COL=0
    readonly TARGET_COL=2
else
    echo No dataset ${DATASET} implemented.
    exit 1;
fi

readonly RESULTS_PATH="/rc_scratch/adwi9965/frankenstein-sweeps/results/${DATASET}-${LANGUAGE}/${ARCH}"

# Print GPU topology if gpus are present
nvidia-smi || true
nvidia-smi topo -m || true

python train_wandb_sweep_agent.py \
       --sweep_id "${SWEEP_ID}" \
       --max_num_runs 1 \
       --arch "${ARCH}" \
       --target_col 2 \
       --features_col "${FEATURES_COL}" \
       --accelerator gpu \
       --experiment "$(date +%s)" \
       --wandb_project "${WANDB_PROJECT}" \
       --train "${TRAIN}" \
       --val "${DEV}" \
       --max_epochs 800 \
       --max_batch_size 256 \
       --patience 50 \
       --save_top_k 1 \
       --check_val_every_n_epoch 4 \
       --model_dir "${RESULTS_PATH}" \
       --seed 42 ;

# Clears wandb cache after each run.
wandb artifact cache cleanup 1gb
