#!/bin/bash

task=sig-2017-inflection
for lang in bulgarian french; do
    for arch in attentive_lstm transformer; do
        python make_wandb_sweep.py \
            --project ${lang}-${task}-${arch}-frankenstein \
            --sweep ${lang}-${task}-${arch}-frankenstein-sweep \
            --outpath SWEEPS.csv
    done
done
