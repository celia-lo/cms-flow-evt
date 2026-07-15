import argparse
import yaml
import os
import re

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import DataLoader
from tqdm import tqdm

from fs_lightning import FlowLightning
from fs_npf_lightning import FlowNumPFLightning

from utils.datasetloader import FastSimDataset

from models.dpm import DPM_Solver, NoiseScheduleFlow


def reshape_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


# BS = 200

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-p", "--checkpoint", type=str, required=True)
parser.add_argument("-ce", "--config_evt", type=str, required=True)
parser.add_argument("-pe", "--checkpoint_evt", type=str, required=True)
parser.add_argument("-n", "--n_steps", type=int, default=40)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-e", "--eval_dir", type=str, default="evals")
parser.add_argument("-ne", "--num_events", type=int, default=100_00)
parser.add_argument("-bs", "--batch_size", type=int, default=200)
parser.add_argument("--test_path", type=str, default=None)
parser.add_argument("--prefix", type=str, default="")
args = parser.parse_args()

with open(args.config, "r") as fp:
    config = yaml.full_load(fp)

ckpt_path = args.checkpoint
epoch = re.search(r"(?<=epoch=).\d*", ckpt_path)
if epoch is None:
    epoch = "last"
else:
    epoch = epoch.group(0)

net = FlowLightning(config)
if os.path.exists(f"{args.eval_dir}") is False:
    os.makedirs(f"{args.eval_dir}")
eval_path = f"{args.eval_dir}/{args.prefix}{config['name']}_{epoch}_{args.n_steps}.root"

checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
for name, param in checkpoint["state_dict"].items():
    if name in net.state_dict():
        try:
            net.state_dict()[name].copy_(param)
        except:
            print(f"Failed to copy {name}")

torch.set_grad_enabled(False)
net.eval()
net.net.eval()
device = torch.device(f"cuda:{args.gpu}")
net.to(device)
net = torch.compile(net)
test_path = config["truth_path_test"] if args.test_path is None else args.test_path
dataset = FastSimDataset(
    test_path,
    config,
    reduce_ds=args.num_events,
    entry_start=0,
    mode="eval",
)

loader = DataLoader(
    dataset,
    num_workers=6,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)

n_events = len(dataset)
l_tr_data = {
    key.replace("pflow_", "").replace("ptrel", "pt"): np.zeros(
        (n_events, dataset.max_particles), dtype=np.float32
    )
    for key in net.pflow_variables + ["ind"]
}
l_fs_data = {
    key.replace("pflow_", "").replace("ptrel", "pt"): np.zeros(
        (n_events, dataset.max_particles), dtype=np.float32
    )
    for key in net.pflow_variables + ["ind"]
}
l_eventNumber = np.zeros(n_events, dtype=np.int32)
n = 0

npf_cfg_path = args.config_evt 
npf_ckpt_path = args.checkpoint_evt

with open(
    npf_cfg_path,
    "r",
) as f:
    npf_cfg = yaml.full_load(f)
npf_model = FlowNumPFLightning(npf_cfg)
npf_model.load_state_dict(
    torch.load(npf_ckpt_path, map_location=torch.device("cpu"), weights_only=False)[
        "state_dict"
    ]
)
npf_model.eval()
npf_model.to(device)
npf_model: FlowNumPFLightning = torch.compile(npf_model)


fs_in_dim = net.net.fs_in_dim
ht_mean, ht_std = (
    dataset.var_transform_dict["ht"].shift,
    dataset.var_transform_dict["ht"].scale,
)
wrong_events_list = []
raw_n = 0
curr_idx = 0


def model_fn(x, timestep, truth, mask, global_data):
    return (1 - timestep.view(-1, 1, 1)) * net.net(
        x, truth, mask, timestep, global_data
    ) + x

sampler = DPM_Solver(
    model_fn=model_fn,
    noise_schedule=NoiseScheduleFlow(),
)

for i, batch in tqdm(enumerate(loader), total=len(loader)):
    truth, truth_mask, global_data = batch
    npf_ext_shape = (truth.shape[0], 4)
    pred = npf_model.sample(
        truth.to(device),
        npf_ext_shape,
        truth_mask.to(device),
        global_data=global_data.to(device),
        method="pndm",
    ).cpu()
    n_pf_pred = (
        npf_model.var_transform_dict["npart"]
        .inverse_transform(pred[..., 1])
        .round()
        .int()
    )
    idxs = np.arange(pred.shape[0])
    pf_ht_pred = pred[..., 0]

    good_idxs = idxs[
        (pf_ht_pred > -ht_mean / ht_std) & (n_pf_pred > 0) & (n_pf_pred < dataset.max_particles)
    ]

    pf_ht_pred[pf_ht_pred < -ht_mean / ht_std] = global_data[..., -1][
        pf_ht_pred < -ht_mean / ht_std
    ]

    n_tr = truth_mask.sum(-1).int()
    n_pf_pred[n_pf_pred < 1] = n_tr[n_pf_pred < 1]
    n_pf_pred[n_pf_pred > dataset.max_particles] = n_tr[n_pf_pred > dataset.max_particles]

    pf_met_x_pred = pred[..., 2]
    pf_met_y_pred = pred[..., 3]

    event_number = dataset.full_data_array["eventNumber"][
        raw_n : truth.shape[0] + raw_n
    ].numpy()

    truth = truth[good_idxs]
    truth_mask = truth_mask[good_idxs]
    global_data = global_data[good_idxs]

    n_pf_pred = n_pf_pred[good_idxs]
    pf_ht_pred = pf_ht_pred[good_idxs]
    pf_met_x_pred = pf_met_x_pred[good_idxs]
    pf_met_y_pred = pf_met_y_pred[good_idxs]

    event_number = event_number[good_idxs]


    raw_n += len(idxs)

    sample_mask = torch.zeros(
        (truth.shape[0], dataset.max_particles, 2), dtype=torch.bool
    )
    sample_mask[..., 0] = truth_mask

    for i in range(n_pf_pred.shape[0]):
        sample_mask[i, : n_pf_pred[i], 1] = True
        sample_mask[i, n_pf_pred[i] :, 1] = False

    truth_ht_scaled = global_data[..., -1]
    global_data = torch.cat(
        [
            global_data[..., :-4],  # scale info
            global_data[..., -4:-2],  # truth_met_x, truth_met_y
            pf_met_x_pred.unsqueeze(-1),  # pf_met_x
            pf_met_y_pred.unsqueeze(-1),  # pf_met_y
            global_data[..., -2:],  # truth_npart, truth_ht
            net.var_transform_dict["npart"]
            .transform(n_pf_pred.float())
            .unsqueeze(-1),  # n_pf_pred
            pf_ht_pred.unsqueeze(-1),  # pf_ht_pred
        ],
        -1,
    )
    truth = truth.to(device)
    truth = torch.cat(
        [
            truth[..., :2],
            torch.sin(truth[..., 2] * 1.814).unsqueeze(-1),
            torch.cos(truth[..., 2] * 1.814).unsqueeze(-1),
            truth[..., 3:],
        ],
        -1,
    )
    fs = (
        sampler.sample(
            torch.randn((*truth.shape[:-1], fs_in_dim)).to(device),
            truth=truth.to(device),
            mask=sample_mask.to(device),
            global_data=global_data.to(device),
            steps=args.n_steps,
            method="multistep",
            skip_type="time_uniform_flow",
            order=2,
        )
        .cpu()
        .to(torch.float32)
    )

    tr_mask = sample_mask[..., 0]
    pflow_ht = net.var_transform_dict["ht"].inverse_transform(pf_ht_pred).numpy()
    truth_ht = net.var_transform_dict["ht"].inverse_transform(truth_ht_scaled).numpy()
    truth = torch.cat(
        [
            truth[..., :2],
            torch.atan2(
                truth[..., 2],
                truth[..., 3],
            ).unsqueeze(-1)
            / 1.814,
            truth[..., 4:],
        ],
        -1,
    )
    truth = truth.to("cpu")
    fs = torch.cat(
        [
            fs[..., :2],
            torch.atan2(
                fs[..., 2],
                fs[..., 3],
            ).unsqueeze(-1)
            / 1.814,
            fs[..., 4:],
        ],
        -1,
    )
    for i, var in enumerate(net.pflow_variables):
        var_name = var.replace("pflow_", "")
        if var_name == "class":
            tr_data_ = truth[..., i:].argmax(-1).cpu().numpy()
            fs_data_ = fs[..., i:].argmax(-1).cpu().numpy()
        else:
            tr_data_ = (
                net.var_transform_dict[var_name]
                .inverse_transform(truth[..., i])
                .cpu()
                .numpy()
            )
            fs_data_ = (
                net.var_transform_dict[var_name]
                .inverse_transform(fs[..., i])
                .cpu()
                .numpy()
            )
            if var_name == "phi":
                tr_data_ = reshape_phi(tr_data_)
                fs_data_ = reshape_phi(fs_data_)
            if var_name == "ptrel":
                tr_data_ = tr_data_ * truth_ht.reshape(-1, 1)
                fs_data_ = fs_data_ * pflow_ht.reshape(-1, 1)
                var_name = "pt"
        l_tr_data[var_name][n : truth.shape[0] + n] = tr_data_
        l_fs_data[var_name][n : truth.shape[0] + n] = fs_data_
    l_tr_data["ind"][n : truth.shape[0] + n] = sample_mask[..., 0].cpu().numpy()
    l_fs_data["ind"][n : truth.shape[0] + n] = sample_mask[..., 1].cpu().numpy()
    l_eventNumber[n : truth.shape[0] + n] = event_number
    n += truth.shape[0]

for key in l_fs_data.keys():
    l_fs_data[key] = l_fs_data[key][:n]
    l_tr_data[key] = l_tr_data[key][:n]
l_eventNumber = l_eventNumber[:n]

with uproot.recreate(eval_path) as file:
    file[f"evt_tree"] = {
        "pflow": ak.zip(
            {
                key: ak.Array(l_fs_data[key].astype(np.float32).tolist())
                for key in l_fs_data.keys()
            }
        ),
        "truth": ak.zip(
            {
                key: ak.Array(l_tr_data[key].astype(np.float32).tolist())
                for key in l_tr_data.keys()
            }
        ),
        "eventNumber": ak.Array(l_eventNumber),
    }

print(f"Saved to {eval_path}")
