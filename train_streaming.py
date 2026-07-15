import argparse
import glob
import math
import os
import resource
import socket
import sys
from pathlib import Path

import comet_ml  # have to import before torch/dgl
import pytorch_lightning
import torch
import yaml
import yaml_include
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import TQDMProgressBar as ProgressBar
from pytorch_lightning.loggers import CometLogger
from lightning.pytorch.profilers import AdvancedProfiler

import utils.fileutils as fu
from fs_lightning import FlowLightning
from fs_lightning_stream import FlowLightningStreaming
from fs_npf_lightning import FlowNumPFLightning


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

os.environ["NCCL_SOCKET_NTHREADS"] = "2"
os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"

yaml.add_constructor(
    "!include", yaml_include.Constructor(base_dir=Path(__file__).parent / "configs")
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Diffusion tagger.")
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("--gpus", default="", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--test_run", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--reduce_dataset", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--no_logging", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def update_config(args, config):
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    return config


def setup_logger(args, config):
    if args.test_run or args.no_logging:
        return None
    if config.get("logger", "").lower() == "none":
        return None

    comet_logger = setup_comet_logger(config, args, os.environ.get("COMET_EXP_ID"))
    if comet_logger.experiment.get_key():
        os.environ["COMET_EXP_ID"] = comet_logger.experiment.get_key()
    return comet_logger


def setup_comet_logger(config, args, exp_id=None):
    comet_logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        save_dir="logs",
        project_name="f_delphes",
        workspace=os.environ["COMET_WORKSPACE"],
        experiment_name=config["name"],
        experiment_key=exp_id,
    )

    if os.environ.get("LOCAL_RANK") is None:
        for key, value in config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    comet_logger.experiment.log_parameter(f"{key}_{k}", v)
            else:
                comet_logger.experiment.log_parameter(key, value)

        comet_logger.experiment.log_parameter("torch_version", torch.__version__)
        comet_logger.experiment.log_parameter(
            "lightning_version", pytorch_lightning.__version__
        )
        comet_logger.experiment.log_parameter("cuda_version", torch.version.cuda)
        comet_logger.experiment.log_parameter("hostname", socket.gethostname())

        comet_logger.experiment.log_asset(args.config)
        all_files = (
            glob.glob("./*.py") + glob.glob("models/*.py") + glob.glob("utils/*.py")
        )
        for fpath in all_files:
            comet_logger.experiment.log_code(fpath)

    return comet_logger


def get_callbacks(config, args):
    refresh_rate = 1 if args.test_run else 20
    callbacks = [ProgressBar(refresh_rate=refresh_rate)]

    assert (config.get("use_ema", False) != config["use_swa"]) or config[
        "use_ema"
    ] is False, "Cannot use both EMA and SWA"

    if not args.test_run:
        monitor_loss = "val_loss_avg"
        file_name = config["run_name"] + "-{epoch:02d}-{" + monitor_loss + ":.4f}"
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_loss,
            dirpath=os.path.join("saved_models/", config["run_name"], "ckpts"),
            filename=file_name,
            save_top_k=-1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    if config["use_swa"]:
        callbacks.append(StochasticWeightAveraging(swa_lrs=float(config["learningrate"])))

    return callbacks


def train(args, config, logger):
    if config.get("train_type", "particle") == "evt":
        lght = FlowNumPFLightning
    elif config.get("streaming_dataset", False):
        lght = FlowLightningStreaming
    else:
        lght = FlowLightning

    model = lght(config)

    if os.environ.get("LOCAL_RANK") is None and logger is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.experiment.log_parameter("trainable_params", trainable_params)

    if args.ckpt_path:
        print("Loading previously trained model from checkpoint:", args.ckpt_path)
        model = lght.load_from_checkpoint(args.ckpt_path, config=config)

    if config["num_gpus"]:
        config["num_workers"] = max(1, config["num_workers"] // config["num_gpus"])

    callbacks = get_callbacks(config, args)

    print("Creating trainer...")
    if config.get("precision", "32-true") != "32":
        torch.set_float32_matmul_precision("medium")
    profiler = (
        AdvancedProfiler(dirpath=".", filename="perf_logs") if args.profile else None
    )

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
        profiler=profiler,
        num_sanity_val_steps=config.get("num_sanity_val_steps", 0),
    )

    print("Fitting model...")
    if args.ckpt_path:
        trainer.fit(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model)

    return model, trainer


def print_job_info(args, config):
    if os.environ.get("LOCAL_RANK") is not None:
        return
    print("-" * 100)
    print("torch", torch.__version__)
    print("lightning", pytorch_lightning.__version__)
    print("cuda", torch.version.cuda)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("Visible GPUs:", args.gpus, " - ", device_name)
    print("-" * 100, "\n")


def parse_gpus(config, gpus):
    num_gpus = len(gpus.split(",")) if gpus != "" else None
    accelerator = "gpu" if num_gpus is not None else "cpu"
    config["accelerator"] = accelerator
    config["num_gpus"] = num_gpus
    return config


def cleanup(config, model):
    print("-" * 100)
    print("Cleaning up...")
    if model.global_rank != 0:
        sys.exit(0)


def shard_dataset_config(config):
    if not config.get("shard_per_rank", False):
        return False
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world <= 1:
        if rank == 0:
            print("[Shard] WORLD_SIZE=1, skipping sharding")
        return False

    for tag, length_key, start_key in (
        ("train", "reduce_ds_train", "entry_start_train"),
        ("valid", "reduce_ds_valid", "entry_start_valid"),
    ):
        total = config.get(length_key)
        if total is None:
            continue
        if isinstance(total, float) and total < 1.0:
            if rank == 0:
                print(f"[Shard] {tag}: fractional reduce_ds, skipping")
            continue
        total = int(total)
        base = config.get(start_key, 0)
        chunk = math.ceil(total / world)
        start = base + chunk * rank
        remaining = total - chunk * rank
        length = min(chunk, max(remaining, 0))
        if length <= 0:
            raise RuntimeError(
                f"Rank {rank} has no data for {tag}. Reduce number of ranks or disable sharding."
            )
        config[start_key] = start
        config[length_key] = length
        print(
            f"[Shard][Rank {rank}/{world}] {tag}: start={start}, len={length}, chunk={chunk}, total={total}"
        )
    return True


def main():
    args = parse_args()
    with open(args.config, "r") as file:
        config = yaml.full_load(file)

    config = update_config(args, config)
    config = parse_gpus(config, args.gpus)
    if config.get("num_nodes") is None:
        config["num_nodes"] = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))

    sharded = shard_dataset_config(config)
    config["_sharded_datasets"] = sharded

    print_job_info(args, config)
    logger = setup_logger(args, config)

    if not args.test_run:
        config = fu.prep_out_dir(args, config)

    if args.test_run:
        factor = 2 * config.get("val_batchsize", config["batchsize"])
        config["reduce_ds_train"] = factor
        config["reduce_ds_valid"] = factor

    model, trainer = train(args, config, logger)

    if not args.test_run:
        cleanup(config, model)


if __name__ == "__main__":
    main()
