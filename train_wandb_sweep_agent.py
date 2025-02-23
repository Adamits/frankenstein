#!/usr/bin/env python

import argparse
import copy
import functools
import math
import traceback
from typing import Tuple

import pytorch_lightning as pl
import wandb
from torchtnt.utils.flops import FlopTensorDispatchMode

from yoyodyne import train, util


class Error(Exception):
    pass


def make_batch_size_and_accumulation(
    batch_size: int, max_batch_size: int
) -> Tuple[int, int]:
    """Calculates a batch size and number of gradient accumulation steps
    s.t. the requested `batch_size` can fit on a device that can only
    compute `max_batch_size` at a time.

    Args:
        batch_size (int): The requested effective batch_size
        max_batch_size (int): The maximum that can fit on the device

    Returns:
        Tuple[int, int]: The actual batch size `b`, and the accumulation
        factor `k`. This tells the trainer to train for `k` steps on a
        batch size `b` before backpropogating.
    """
    if batch_size <= max_batch_size:
        return batch_size, 1

    # Otherwise we want the smallest k s.t. batch_size can be evenly split
    factor = math.ceil(batch_size / max_batch_size)
    new_batch_size = batch_size / factor

    while not new_batch_size.is_integer():
        factor += 1
        new_batch_size = batch_size / factor

    return int(new_batch_size), factor


def run_train(args):
    # First get trainer to initialize the wandb run
    trainer = train.get_trainer_from_argparse_args(args)
    pl.seed_everything(args.seed)

    batch_size, accumulation_steps = make_batch_size_and_accumulation(
        wandb.config.batch_size, args.max_batch_size
    )
    # Adds accumulate_grad_batches to the trainer
    args.accumulate_grad_batches = accumulation_steps
    util.log_info(
        f"Using batch_size={batch_size} with {accumulation_steps} "
        "accumulation steps for the requested effective batch_size"
        f" of {wandb.config.batch_size}"
    )
    # Model args come from the W&B sweep config.
    if wandb.config.scheduler == "warmupinvsqrt":
        # Computes warmup steps as a function of num_warmup_samples, and the
        # batch size *in practice* -- NOT the one specified in the wandb.config
        # This is because we use graient accumulation over multiple steps.
        warmup_steps = int(wandb.config.num_warmup_samples / batch_size)
        util.log_info(f"Running {warmup_steps} warmup steps")
    else:
        warmup_steps = 0

    # Model args come from the W&B sweep config.
    kwargs = dict(wandb.config)
    # Anything not specified in the config is taken from the CLI args.
    kwargs.update({k: v for k, v in vars(args).items() if k not in kwargs})
    # Manually override batch_size and warmup steps based on
    # the calculations done above.
    kwargs["batch_size"] = batch_size
    kwargs["warmup_steps"] = warmup_steps
    # Updates num epochs and patience depending on arch, and LR.
    # This is to reduce training time for LSTMs that have a learning rate we know should
    # converge relatively fast.
    if kwargs["arch"] == "attentive_lstm" or kwargs["arch"] == "lstm":
        if kwargs["learning_rate"] > 1e-4:
            util.log_info("Detected LSTM with sufficiently high LR")
            util.log_info("Reducing max_epochs to 600 and patience to 15")
            kwargs["max_epochs"] = 600
            kwargs["patience"] = 20
    
    #kwargs["reduceonplateau_mode"] = "loss"
    #kwargs["reduceonplateau_factor"] = kwargs["factor"]
    #kwargs["reduceonplateau_patience"] = kwargs["reduce_lr_patience"]
    new_args = argparse.Namespace()
    # TODO: This is real hacky.
    # I think that this points to the vals in args, and makes
    # them assignable as a dict.
    new_args_vars = vars(new_args)
    for k, v in kwargs.items():
        new_args_vars[k] = v

    datamodule = train.get_datamodule_from_argparse_args(new_args)
    model = train.get_model_from_argparse_args(new_args, datamodule)

    with FlopTensorDispatchMode(model) as ftdm:
        # NOTE: Doing this here will change the order 
        #       of the dataset in the first epoch for actual training.
        x = next(iter(datamodule.train_dataloader()))
        # TODO: This calls forward and backward
        # Instead, we can compute just forward, then 
        # simulate computing a loss and calling backward
        # then separately compute backwards FLOPs?
        res = model.training_step(x, 0, pack_sequences=False)
        flops = copy.deepcopy(ftdm.flop_counts)
        ftdm.reset()
        # NOTE: I think this has sums of the flops in all subcomponents
        # TODO: Not sure if this works right nor if I interpret it rught
        #       did not see much by way of documentation on torchtnt
        flops_dict = flops[""]
        all_flops = sum(flops_dict.values())
        if all_flops > 1000000:
            all_flops = f"{all_flops // 1000000}M"
        elif all_flops > 1000000000:
            all_flops = f"{all_flops // 1000000000}B"
        util.log_info(f"FLOPs from one train step: {all_flops}")
        # Ensure no gradients when we actually start training.
        model.zero_grad()
    # Logs number of model parameters to W&B.
    if args.log_wandb:
        wandb.config["n_model_params"] = sum(
            p.numel() for p in model.parameters()
        )
        wandb.config["FLOPs"] = flops_dict
        wandb.config["total_FLOPs"] = sum(flops_dict.values())

    # Train and log the best checkpoint.
    best_checkpoint = train.train(
        trainer, model, datamodule, args.train_from
    )
    util.log_info(f"Best checkpoint: {best_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    train.add_argparse_args(parser)
    parser.add_argument(
        "--sweep_id",
        help="ID for the sweep to run the agent in.",
    )
    parser.add_argument(
        "--wandb_project",
        help="project name for the sweep to run the agent in.",
    )
    parser.add_argument(
        "--max_num_runs",
        type=int,
        default=1,
        help="Max number of runs this agent should train.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Max number samples in a batch on device."
        " Will be used to compute the batch size in practice, but we use"
        " gradient accumulation so as not to run OOM, but keep the"
        " requested batch size.",
    )
    args = parser.parse_args()
    # Forces log_wandb to True, so that the PTL trainer logs
    # runtime metrics to W&B
    args.log_wandb = True
    try:
        wandb.agent(
            args.sweep_id,
            function=functools.partial(run_train, args),
            project=args.wandb_project,
            count=args.max_num_runs,
        )
    except Exception:
        # Exits gracefully, so wandb logs the error
        util.log_info(traceback.format_exc())
        exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
