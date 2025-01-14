import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.utils import load_configs, initialize_logging, get_scheduler, get_optimizer
import wandb
import argparse
from loader.Dataloader import load_data_from_npy, load_split_data
from torch.utils.data import DataLoader
from layer.models import CFNO


def parse_args():
    parser = argparse.ArgumentParser(description='CFNO')
    parser.add_argument('--config', type=str, default='./configs/configs.yaml',
                        help='Path to the config file')

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)





if __name__ == "__main__":
    args = parse_args()
    set_seed(42)
    # Load configs, initialize logging, and set seed
    configs = load_configs(file_path=args.config)
    seed = configs.get('seed', 42)
    set_seed(seed)
    logger = initialize_logging(file_dir="./logs")
    logger.info(f"Loaded configs from: {args.config}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Configs: {configs}")


    if configs.get('use_wandb', False):
        wandb.init(project="CFNO", config=configs)
        config = wandb.config
    else:
        config = configs

    # load and split data
    train_dataset, test_dataset = load_split_data(config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    logger.info(f"Loaded data from: {config['data_dir']}")

    # Define the model
    m = train_dataset.m
    n = train_dataset.n
    T = train_dataset.T

    model = CFNO(n, m, T, config, logger)
    logger.info(f"Model: {model}")

    optimizer = get_optimizer(model, config, logger)
    scheduler = get_scheduler(optimizer, config, logger)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")
    
    # Train the model
    
