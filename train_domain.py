"""Training script for ALEPH→DELPHI reco domain translation models.

Usage:
  # Particle model
  python train_domain.py -c configs/domain_part.yaml --gpus 0

  # Event model
  python train_domain.py -c configs/domain_evt.yaml --gpus 0

  # Resume from checkpoint
  python train_domain.py -c configs/domain_part.yaml --gpus 0 \
    --ckpt_path saved_models/<run_name>/ckpts/last.ckpt

  # Quick sanity check (no logging, no checkpoints)
  python train_domain.py -c configs/domain_part.yaml --gpus 0 --test_run

Reuses all boilerplate from train.py (parse_args, get_callbacks, etc.).
Only the model selection differs: domain_particle → DomainTranslationLightning,
domain_evt → DomainEvtLightning.
"""

import os
import sys
from pathlib import Path

import torch
import yaml
import yaml_include
from pytorch_lightning import Trainer

import utils.fileutils as fu
from domain_evt_lightning import DomainEvtLightning
from domain_translation_lightning import DomainTranslationLightning

# Reuse all boilerplate helpers from train.py
from train import (
    cleanup,
    get_callbacks,
    parse_args,
    parse_gpus,
    print_job_info,
    setup_logger,
    update_config,
)

yaml.add_constructor(
    "!include", yaml_include.Constructor(base_dir=Path(__file__).parent / "configs")
)


def train_domain(args, config, logger):
    train_type = config.get("train_type", "domain_particle")

    if train_type == "domain_particle":
        lght_cls = DomainTranslationLightning
    elif train_type == "domain_evt":
        lght_cls = DomainEvtLightning
    else:
        raise ValueError(
            f"Unknown train_type '{train_type}'. "
            "Expected 'domain_particle' or 'domain_evt'."
        )

    model = lght_cls(config)

    if os.environ.get("LOCAL_RANK") is None and logger is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.experiment.log_parameter("trainable_params", trainable_params)

    if args.ckpt_path:
        print("Loading from checkpoint:", args.ckpt_path)
        model = lght_cls.load_from_checkpoint(args.ckpt_path, config=config)

    if config["num_gpus"]:
        config["num_workers"] = config["num_workers"] // config["num_gpus"]

    callbacks = get_callbacks(config, args)

    if config.get("precision", "32-true") != "32":
        torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        max_epochs=config["num_epochs"],
        accelerator=config["accelerator"],
        devices=config["num_gpus"],
        num_nodes=config.get("num_nodes", 1),
        logger=logger,
        log_every_n_steps=20,
        fast_dev_run=args.test_run,
        callbacks=callbacks,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        gradient_clip_val=config.get("gradient_clip_val", None),
        precision=config.get("precision", "32-true"),
        limit_val_batches=config.get("limit_val_batches", 1.0),
    )

    print("Fitting model...")
    if args.ckpt_path:
        trainer.fit(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model)

    return model, trainer


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.full_load(f)

    config = update_config(args, config)
    config = parse_gpus(config, args.gpus)
    if config.get("num_nodes") is None:
        config["num_nodes"] = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))

    print_job_info(args, config)

    logger = setup_logger(args, config)

    if not args.test_run:
        config = fu.prep_out_dir(args, config)

    if args.test_run:
        small = 2 * config.get("val_batchsize", config["batchsize"])
        config["reduce_ds_source_train"] = small
        config["reduce_ds_source_valid"] = small
        config["reduce_ds_target_train"] = small
        config["reduce_ds_target_valid"] = small

    model, trainer = train_domain(args, config, logger)

    if not args.test_run:
        cleanup(config, model)


if __name__ == "__main__":
    main()
