"""Creates a hyperparameter sweep in the wandb API.

Manually modify functions below with different Dicts to change
the hyperparameters, or sweep method (defaults to random).

See here for details:
https://docs.wandb.ai/guides/sweeps/define-sweep-configuration"""

import argparse
from typing import Dict
import os

import wandb

# Turn off the wandb logs, so we can log only the sweep id.
os.environ['WANDB_SILENT']="true"


def get_hparams() -> Dict:
    """Gets the dictionary of hyperparams to sweep.

    Returns:
        Dict: Dictionary of hyperparameter names and value distributions.
    """
    # Optimization params
    optim_hparams = {
        "batch_size": {
            "distribution": "q_uniform",
            "q": 16,
            "min": 16,
            "max": 2048,
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.00002,
            "max": 0.002,
        },
        "label_smoothing": {"distribution": "uniform", "min": 0.0, "max": 0.2},
        "scheduler": {"values": ["warmupinvsqrt", None]},
        # TODO: do we want a smaller range?
        # NOTE: This is irrelevent if None scheduler is sampled.
        "num_warmup_samples": {
            "distribution": "q_uniform",
            "q": 100,
            "min": 0,
            "max": 5000000,
        },
    }

    # Hyperparameters that impact the actual architecture.
    arch_hparams = {
        "embedding_size": {
            "distribution": "q_uniform",
            "q": 64,
            "min": 16,
            "max": 1024,
        },
        "hidden_size": {
            "distribution": "q_uniform",
            "q": 128,
            "min": 64,
            "max": 2048,
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0,
            "max": 0.5,
        },
        "attention_heads": {"values": [2, 4, 8]}, # NOTE: this is meaningless for LSTM
        # TODO: Should we constrain sampling to 2, 4, 6, 8?
        "encoder_layers": {
            "distribution": "q_uniform",
            "q": 1,
            "min": 1,
            "max": 8,
        },
        "decoder_layers": {
            "distribution": "q_uniform",
            "q": 1,
            "min": 1,
            "max": 8,
        },
    }
    # Combines the optimization and architectural params into one dict.
    return optim_hparams | arch_hparams


def make_sweep(project: str, sweep: str) -> int:
    """Creates the sweep in the wandb API, according to the hyperparameter
    ranges in `HPARAMS`.

    Args:
        project (str): Name of the wandb project.
        sweep (str): Name of the wandb sweep.

    Returns:
        int: The wandb sweep ID for this configuration.
    """
    # TODO: add early stopping.
    sweep_configuration = {
        "method": "random",#"bayes",
        "name": sweep,
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": get_hparams(),
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    return sweep_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Name of the project")
    parser.add_argument("--sweep", required=True, help="Name of the sweep")
    parser.add_argument(
        "--outpath", required=True, help="Path to append sweep info to."
    )
    args = parser.parse_args()
    sweep_id = make_sweep(args.project, args.sweep)

    with open(args.outpath, "a") as out:
        print(f"{args.project},{args.sweep},{sweep_id}", file=out)


if __name__ == "__main__":
    main()
