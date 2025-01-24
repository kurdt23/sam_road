from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config
from dataset_semi import SatMapDataset, graph_collate_fn
from model_semi import SAMRoad

import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
         "If None, use the nnU-Net config. See /config for examples.",)
parser.add_argument(
    "--osm_masks_dir", default=None, help="Path to the directory with OpenStreetMap masks")
parser.add_argument(
    "--inference_masks_dir", default=None, help="Path to the directory with inference masks")
parser.add_argument(
    "--resume", default=None, help="checkpoint of the last epoch of the model")
parser.add_argument(
    "--precision", default=16, help="32 or 16")
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true')
parser.add_argument(
    "--dev_run", default=False, action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    use_semi_supervised = config.get('USE_SEMI_SUPERVISED', False)
    dev_run = args.dev_run or args.fast_dev_run

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="sam_road",
        # track hyperparameters and run metadata
        config=config,
        # disable wandb if debugging
        mode='disabled' if dev_run else None
    )

    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    net = SAMRoad(config)

    # train_ds, val_ds = SatMapDataset(config, is_train=True, dev_run=dev_run), SatMapDataset(config, is_train=False,
    #                                                                                         dev_run=dev_run)

    # OPEN STREET MASK
    train_ds = SatMapDataset(config, is_train=True, dev_run=dev_run, osm_masks_dir=args.osm_masks_dir,
                             use_semi_supervised=use_semi_supervised)
    val_ds = SatMapDataset(config, is_train=False, dev_run=dev_run, osm_masks_dir=args.osm_masks_dir,
                           use_semi_supervised=use_semi_supervised)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger()

    # from lightning.pytorch.profilers import AdvancedProfiler
    # profiler = AdvancedProfiler(dirpath='profile', filename='result_fast_matcher')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        # profiler=profiler
    )

    # trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)
