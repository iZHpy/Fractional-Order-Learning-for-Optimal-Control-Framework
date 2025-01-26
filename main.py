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
from layer.models import CFNO, baseFNO
from utils.utils import LpLoss, UnitGaussianNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description='CFNO')
    parser.add_argument('--config', type=str, default='./configs/configs.yaml',
                        help='Path to the config file')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def train(model, train_loader, train_norms, optimizer, device, logger):
    model.train()
    total_loss = 0
    total_regression_loss = 0
    total_label_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {key: value.to(device) for key, value in batch.items()}
        _, loss, regression_loss, label_loss = model(batch, train_norms, logger)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_regression_loss += regression_loss.item()
        total_label_loss += label_loss.item()
    logger.info(f"Train Loss: {total_loss / len(train_loader)}")
    logger.info(f"Train Regression Loss: {total_regression_loss / len(train_loader)}")
    logger.info(f"Train Label Loss: {total_label_loss / len(train_loader)}")
    return total_loss / len(train_loader)

def test(model, test_loader, test_norms, device, logger):
    model.eval()
    total_loss = 0
    total_regression_loss = 0
    total_label_loss = 0
    ys_pred = []
    ys_true = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            y_pred, loss, regression_loss, label_loss = model(batch, test_norms, logger)
            ys_pred.append(y_pred)
            if test_norms is not None:
                ys_true.append(test_norms['optimal_controls'].decode(batch['optimal_controls']))
            else:
                ys_true.append(batch['optimal_controls'])
            total_loss += loss.item()
            total_regression_loss += regression_loss.item()
            total_label_loss += label_loss.item()

    logger.info(f"Test Loss: {total_loss / len(test_loader)}")
    logger.info(f"Test Regression Loss: {total_regression_loss / len(test_loader)}")
    logger.info(f"Test Label Loss: {total_label_loss / len(test_loader)}")
    metrics(torch.cat(ys_pred, dim=0), torch.cat(ys_true, dim=0), logger)
    return total_loss / len(test_loader)

def metrics(ys_pred, ys_true, logger):
    mse = F.mse_loss(ys_pred, ys_true)
    mae = F.l1_loss(ys_pred, ys_true)
    lploss = LpLoss()(ys_pred, ys_true)
    logger.info(f"MSE: {mse}")
    logger.info(f"MAE: {mae}")
    logger.info(f"LpLoss: {lploss}")
    if mse < best_metric[0]:
        best_metric[0] = mse
        best_epoch[0] = epoch
    if mae < best_metric[1]:
        best_metric[1] = mae
        best_epoch[1] = epoch
    if lploss < best_metric[2]:
        best_metric[2] = lploss
        best_epoch[2] = epoch

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # load and split data
    train_dataset, test_dataset, train_norms, test_norms = load_split_data(config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    logger.info(f"Loaded data from: {config['data_dir']}")
    # Define the model
    m = train_dataset.m
    n = train_dataset.n
    T = train_dataset.T

    model = CFNO(n, m, T, config, logger).to(device)
    # model = baseFNO(n, m, T, config, logger).to(device)
    if train_norms is not None:
        train_norms['optimal_controls'].cuda(device)
    if test_norms is not None:
        test_norms['optimal_controls'].cuda(device)
        logger.info("Using normalization")

    logger.info(f"Model: {model}")

    optimizer = get_optimizer(model, config, logger)
    scheduler = get_scheduler(optimizer, config, logger)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")
    

    best_metric = [float('inf'), float('inf'), float('inf')]
    best_epoch = [None, None, None]
    # Train the model
    logger.info("Training the model")
    epochs = config['epochs']
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        loss = train(model, train_loader, train_norms, optimizer, device, logger)
        scheduler.step(loss)
        # Evaluate the model
        test_loss = test(model, test_loader, test_norms, device, logger)

    logger.info(f"Best MSE: {best_metric[0]} at epoch {best_epoch[0]}")
    logger.info(f"Best MAE: {best_metric[1]} at epoch {best_epoch[1]}")
    logger.info(f"Best LpLoss: {best_metric[2]} at epoch {best_epoch[2]}")
    
