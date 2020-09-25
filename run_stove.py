"""Train and evaluate STOVE."""
import sys

from model.main import main

import wandb
import hydra
import numpy as np
import torch

import sys
from omegaconf import OmegaConf, DictConfig
import datetime
import os

if __name__ == '__main__':

    # call script with  --args config_option config_value
    # e.g: pythono main.py --args batch_size 20 encoder cnn plot_every 1
    # sys.argv[1:] = [arg[2:] if arg.startswith('--') else arg for arg in sys.argv[1:]]
    # init for W&B (only if you run `wandb login <token>` before)
    torch.multiprocessing.set_start_method('spawn')
    wandb.init(
        project='lingtest',     #  project name
        name='StoveRun-' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    )

    keys = sys.argv[2::2]
    values = sys.argv[3::2]
    kvdict = dict(zip(keys, values))
    trainer = main(sh_args=kvdict)
    trainer.train()
