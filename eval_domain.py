"""Inference script for ALEPH→DELPHI reco domain translation.

Loads ALEPH reco events, runs the event model to predict DELPHI event properties,
then runs the particle model to generate DELPHI-like particle clouds.
Saves results to a ROOT file in the same evals/ format as eval.py.

Usage:
  python eval_domain.py \\
    -c  configs/domain_part.yaml  -p  saved_models/<part_run>/ckpts/last.ckpt \\
    -ce configs/domain_evt.yaml   -pe saved_models/<evt_run>/ckpts/last.ckpt  \\
    --source_path my_sample/QQB/test_evt.root \\
    -ne 5000 -bs 200 -n 40 --prefix aleph2delphi_

Output:
  evals/<prefix><part_config_name>_<epoch>_<n_steps>.root

  ROOT tree "evt_tree" with branches:
    source/  : ALEPH reco particles (encoded then decoded back to physical)
    generated/: generated DELPHI-like particles
    eventNumber: sequential int
"""

import argparse
import os
import re

import awkward as ak
import numpy as np
import torch
import uproot
import yaml
import yaml_include
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

yaml.add_constructor(
    "!include", yaml_include.Constructor(base_dir=Path(__file__).parent / "configs")
)

from domain_evt_lightning import DomainEvtLightning
from domain_translation_lightning import DomainTranslationLightning
from utils.domain_datasetloader import DomainRecoDataset

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Domain translation inference")
parser.add_argument("-c", "--config", required=True, help="Particle model config YAML")
parser.add_argument("-p", "--checkpoint", required=True, help="Particle model checkpoint")
parser.add_argument("-ce", "--config_evt", required=True, help="Event model config YAML")
parser.add_argument("-pe", "--checkpoint_evt", required=True, help="Event model checkpoint")
parser.add_argument("--source_path", default=None, help="ALEPH source ROOT file")
parser.add_argument("-ne", "--num_events", type=int, default=10_000)
parser.add_argument("-bs", "--batch_size", type=int, default=200)
parser.add_argument("-n", "--n_steps", type=int, default=40)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-e", "--eval_dir", type=str, default="evals")
parser.add_argument("--prefix", type=str, default="")
args = parser.parse_args()

# -------------------------------------------------------------------
# Load configs and models
# -------------------------------------------------------------------

with open(args.config) as f:
    part_cfg = yaml.full_load(f)
with open(args.config_evt) as f:
    evt_cfg = yaml.full_load(f)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Particle model
part_model = DomainTranslationLightning(part_cfg)
part_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
part_model.load_state_dict(part_ckpt["state_dict"])
part_model.eval().to(device)
part_model = torch.compile(part_model)

# Event model
evt_model = DomainEvtLightning(evt_cfg)
evt_ckpt = torch.load(args.checkpoint_evt, map_location="cpu", weights_only=False)
evt_model.load_state_dict(evt_ckpt["state_dict"])
evt_model.eval().to(device)
evt_model = torch.compile(evt_model)

# Epoch tag for output filename
epoch = re.search(r"(?<=epoch=)\d+", args.checkpoint)
epoch = epoch.group(0) if epoch else "last"

os.makedirs(args.eval_dir, exist_ok=True)
eval_path = (
    f"{args.eval_dir}/{args.prefix}{part_cfg['name']}_{epoch}_{args.n_steps}.root"
)

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

source_path = args.source_path or part_cfg.get("source_path_test", part_cfg["source_path_train"])
dataset = DomainRecoDataset(
    source_path, part_cfg, reduce_ds=args.num_events, entry_start=0
)
loader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=4, pin_memory=False
)

n_events = len(dataset)
max_p = part_cfg["max_particles"]
pflow_variables = ["ptrel", "eta", "phi", "class"]  # decoded output variables

# Storage arrays
src_arrays = {v: np.zeros((n_events, max_p), dtype=np.float32) for v in pflow_variables}
gen_arrays = {v: np.zeros((n_events, max_p), dtype=np.float32) for v in pflow_variables}
src_mask_arr = np.zeros((n_events, max_p), dtype=bool)
gen_mask_arr = np.zeros((n_events, max_p), dtype=bool)

ht_vt = part_model.var_transform_dict["ht"]
n_vt = part_model.var_transform_dict["npart"]

# -------------------------------------------------------------------
# Inference loop
# -------------------------------------------------------------------

n_written = 0
with torch.no_grad():
    for batch in tqdm(loader, desc="Generating"):
        src_reco, src_mask, src_global = batch
        bs = src_reco.shape[0]

        src_reco = src_reco.to(device)
        src_mask = src_mask.to(device)
        src_global = src_global.to(device)

        # Step 1: predict DELPHI event properties
        evt_pred = evt_model.sample(src_reco, src_mask, src_global).cpu()
        # evt_pred: [bs, 4] = [Ht_norm, n_norm, MET_x_norm, MET_y_norm]

        n_tgt_pred = n_vt.inverse_transform(evt_pred[:, 1]).round().int()
        ht_tgt_pred_norm = evt_pred[:, 0]  # normalised Ht, for later pt recovery

        # Fallback: clamp invalid predictions to source count
        src_n = src_mask.sum(-1).int().cpu()
        bad = (n_tgt_pred < 1) | (n_tgt_pred >= max_p)
        n_tgt_pred[bad] = src_n[bad]
        ht_tgt_pred_norm[bad] = src_global[:, 0].cpu()[bad]

        # Step 2: generate DELPHI-like particles
        fs, tgt_mask = part_model.sample(
            src_reco, src_mask, n_tgt_pred.to(device),
            src_global, n_steps=args.n_steps,
        )
        fs = fs.cpu()
        tgt_mask = tgt_mask.cpu()
        src_reco = src_reco.cpu()
        src_mask = src_mask.cpu()
        src_global = src_global.cpu()

        # -----------------------------------------------------------
        # Decode source (ALEPH) reco
        # -----------------------------------------------------------
        src_ht = ht_vt.inverse_transform(src_global[:, 0])  # [bs]

        src_ptrel = part_model.var_transform_dict["ptrel"].inverse_transform(src_reco[..., 0])
        src_eta = part_model.var_transform_dict["eta"].inverse_transform(src_reco[..., 1])
        # sin/cos → phi
        src_phi = torch.atan2(src_reco[..., 2], src_reco[..., 3]) / 1.814
        src_cls = src_reco[..., 4:].argmax(-1).float()

        src_pt = src_ptrel * src_ht.unsqueeze(-1)

        # -----------------------------------------------------------
        # Decode generated (DELPHI-like) particles
        # -----------------------------------------------------------
        gen_ht = ht_vt.inverse_transform(ht_tgt_pred_norm)  # [bs]

        gen_ptrel = part_model.var_transform_dict["ptrel"].inverse_transform(fs[..., 0])
        gen_eta = part_model.var_transform_dict["eta"].inverse_transform(fs[..., 1])
        gen_phi = torch.atan2(fs[..., 2], fs[..., 3]) / 1.814
        gen_cls = fs[..., 4:].argmax(-1).float()

        gen_pt = gen_ptrel * gen_ht.unsqueeze(-1)

        # -----------------------------------------------------------
        # Write to storage
        # -----------------------------------------------------------
        end = n_written + bs
        src_arrays["ptrel"][n_written:end] = src_ptrel.numpy()
        src_arrays["eta"][n_written:end] = src_eta.numpy()
        src_arrays["phi"][n_written:end] = src_phi.numpy()
        src_arrays["class"][n_written:end] = src_cls.numpy()
        src_mask_arr[n_written:end] = src_mask.numpy()

        gen_arrays["ptrel"][n_written:end] = gen_ptrel.numpy()
        gen_arrays["eta"][n_written:end] = gen_eta.numpy()
        gen_arrays["phi"][n_written:end] = gen_phi.numpy()
        gen_arrays["class"][n_written:end] = gen_cls.numpy()
        gen_mask_arr[n_written:end] = tgt_mask.numpy()

        n_written = end

# Trim to actual number of events processed
for v in pflow_variables:
    src_arrays[v] = src_arrays[v][:n_written]
    gen_arrays[v] = gen_arrays[v][:n_written]
src_mask_arr = src_mask_arr[:n_written]
gen_mask_arr = gen_mask_arr[:n_written]

# -------------------------------------------------------------------
# Save ROOT file
# -------------------------------------------------------------------

def arrays_to_ak(data_dict, mask):
    """Convert padded arrays to ragged awkward arrays using mask."""
    result = {}
    for k, arr in data_dict.items():
        ragged = [row[m] for row, m in zip(arr, mask)]
        result[k] = ak.Array(ragged)
    return result

with uproot.recreate(eval_path) as f:
    f["evt_tree"] = {
        "source": ak.zip(arrays_to_ak(src_arrays, src_mask_arr)),
        "generated": ak.zip(arrays_to_ak(gen_arrays, gen_mask_arr)),
        "eventNumber": ak.Array(np.arange(n_written, dtype=np.int32)),
    }

print(f"Saved {n_written} events to {eval_path}")
