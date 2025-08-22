import gc
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import uproot
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


class VarTransform:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.type = self.config.get("type", "std")
        self.do_pes = self.config.get("do_pes", False)
        self.fn = self.config.get("fn", None)

        if self.type == "std":
            self.shift = self.config.get("mean", 0.0)
            self.scale = self.config.get("std", 1.0)
        elif self.type == "minmax":
            self.shift = self.config.get("min", 0.0)
            self.scale = self.config.get("max", 1.0) - self.shift

        assert self.type in ["std", "minmax"]

    def calculate(self, x):
        if self.type == "std":
            mean = x.mean()
            if len(x) < 2:
                if self.name in ["eta", "phi"]:
                    std = torch.tensor(0.1).float()
                else:
                    std = torch.tensor(1).float()
            else:
                std = x.std()
                if std == 0:
                    std = (
                        torch.tensor(0.1).float()
                        if self.name in ["eta", "phi"]
                        else torch.tensor(1).float()
                    )
            return mean, std
        elif self.type == "minmax":
            min_, max_ = x.min(), x.max()
            if min_ == max_:
                min_, max_ = min_ - 1, max_ + 1
            return min_, max_ - min_

    def transform(self, x, shift=None, scale=None):
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        if self.fn == "log":
            x = torch.log(x)
        elif self.fn == "log1p":
            x = torch.log1p(x)

        return (x - shift) / scale

    def inverse_transform(self, x, shift=None, scale=None):
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = x * scale + shift
        if self.fn == "log":
            return torch.exp(x)
        elif self.fn == "log1p":
            return torch.expm1(x)
        return x


def normalize(inp):
    return torch.arctan2(torch.sin(inp), torch.cos(inp))


def do_padding(tensor, max_len):
    shape = tensor.shape
    new_shape = (max_len,) + shape[1:]
    x = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    x[: shape[0]] = tensor
    return x


class FastSimDataset(Dataset):
    def __init__(
        self, filename, config=None, reduce_ds=1.0, entry_start=0, mode="train"
    ):
        super().__init__()
        self.config = config
        self.mode = mode

        assert self.mode in ["train", "eval"], "Mode must be either 'train' or 'eval'"

        self.train_type = self.config.get("train_type", "particle")
        self.use_scale_info = self.config.get("use_scale_info", True)
        self.sin_cos = self.config.get("sin_cos", False)

        self.zero_neutral_vtx = self.config.get("zero_neutral_vtx", False)

        self.file = uproot.open(filename, num_workers=6)
        self.tree = self.file[f"evt_tree"]

        self.max_particles = self.config["max_particles"]

        self.entry_start = entry_start
        self.nevents = self.tree.num_entries
        if reduce_ds < 1.0 and reduce_ds > 0:
            self.nevents = int(self.nevents * reduce_ds)
        if reduce_ds >= 1.0:
            self.nevents = reduce_ds
        print(" we have ", self.nevents, " events")

        self.n_particle_mask = None
        self.full_data_array = {}

        self._load_truth()

        if self.mode == "train":
            self._load_pflow()

        self._data_to_tensor(mask_events=True)

        if self.zero_neutral_vtx and self.mode == "train":
            for var in ["vx", "vy", "vz"]:
                self.full_data_array[f"pflow_{var}"][
                    self.full_data_array["pflow_class"] > 2
                ] = 0

        self.full_data_array["eventNumber"] = torch.tensor(
            self.tree["eventNumber"].array(
                library="np",
                entry_stop=self.nevents + self.entry_start,
                entry_start=self.entry_start,
            )[self.n_particle_mask]
        )

        self.n_truth_particles = self.n_truth_particles[self.n_particle_mask]
        self.truth_cumsum = np.cumsum([0] + list(self.n_truth_particles))

        if self.mode == "train":
            self.n_pflow_particles = self.n_pflow_particles[self.n_particle_mask]
            self.pflow_cumsum = np.cumsum([0] + list(self.n_pflow_particles))

        self.file.close()

        if self.train_type == "evt":
            for key in ["pflow_ptrel", "pflow_eta", "pflow_phi", "pflow_class"]:
                self.full_data_array.pop(key)
        self.nevents = len(self.n_truth_particles)
        del self.tree
        gc.collect()
        print("done loading data")

        self.scale_map = {}
        self.var_transform_dict = {
            key: VarTransform(key, val)
            for key, val in self.config["var_transform"].items()
        }

        self._calculate_mean_std()
        self._get_scaled_global_data()

    def _calculate_mean_std(self):
        self.truth_vars_shift_scales = {
            key: torch.zeros(self.nevents, 2) for key in self.truth_variables
        }
        for idx in tqdm(range(self.nevents), desc="Calculating mean and std"):
            truth_start, truth_end = self.truth_cumsum[idx], self.truth_cumsum[idx + 1]
            for key in self.truth_variables:
                var_name = key.replace("truth_", "")
                if var_name == "class":
                    continue
                var_transform = self.var_transform_dict[var_name]
                shift, scale = var_transform.calculate(
                    self.full_data_array[key][truth_start:truth_end]
                )
                self.truth_vars_shift_scales[key][idx] = torch.tensor([shift, scale])

    def _get_scaled_global_data(self):
        if self.train_type == "particle" and self.mode == "train":
            self.scaled_global_data = torch.stack(
                [
                    self.var_transform_dict["met_x"].transform(
                        self.full_data_array["truth_met_x"]
                    ),
                    self.var_transform_dict["met_y"].transform(
                        self.full_data_array["truth_met_y"]
                    ),
                    self.var_transform_dict["met_x"].transform(
                        self.full_data_array["pflow_met_x"]
                    ),
                    self.var_transform_dict["met_y"].transform(
                        self.full_data_array["pflow_met_y"]
                    ),
                    self.var_transform_dict["npart"].transform(
                        torch.tensor(self.n_truth_particles)
                    ),
                    self.var_transform_dict["ht"].transform(
                        self.full_data_array["truth_ht"]
                    ),
                    self.var_transform_dict["npart"].transform(
                        torch.tensor(self.n_pflow_particles)
                    ),
                    self.var_transform_dict["ht"].transform(
                        self.full_data_array["pflow_ht"]
                    ),
                ],
                -1,
            ).float()
        elif self.train_type == "evt" or self.mode == "eval":
            self.scaled_global_data = torch.stack(
                [
                    self.var_transform_dict["met_x"].transform(
                        self.full_data_array["truth_met_x"]
                    ),
                    self.var_transform_dict["met_y"].transform(
                        self.full_data_array["truth_met_y"]
                    ),
                    self.var_transform_dict["npart"].transform(
                        torch.tensor(self.n_truth_particles)
                    ),
                    self.var_transform_dict["ht"].transform(
                        self.full_data_array["truth_ht"]
                    ),
                ],
                -1,
            ).float()

    def _data_to_tensor(self, mask_events=True):
        """
        Converts the data to torch tensors and applies the mask if needed
        """
        for var, value in tqdm(self.full_data_array.items()):
            if mask_events:
                value = value[self.n_particle_mask]
            if "ht" not in var and "met" not in var and "_class_" not in var:
                value = np.concatenate(value)
            value = torch.tensor(value)
            if "eta" in var:
                value = torch.clamp(value, -3, 3)
            elif "phi" in var:
                value = normalize(value)
            self.full_data_array[var] = value

    def _load_truth(self):
        self.truth_variables = [el for el in self.config["truth_variables"]]

        self.n_truth_particles = self.tree["ntruth"].array(
            library="np",
            entry_stop=self.nevents + self.entry_start,
            entry_start=self.entry_start,
        )
        if self.n_particle_mask is None:
            self.n_particle_mask = self.n_truth_particles < self.max_particles
        else:
            self.n_particle_mask = self.n_particle_mask & (
                self.n_truth_particles < self.max_particles
            )

        for var in tqdm(self.truth_variables):
            self.full_data_array[var] = self.tree[var].array(
                library="np",
                entry_stop=self.nevents + self.entry_start,
                entry_start=self.entry_start,
            )
            if var == "truth_pt":
                self.full_data_array["truth_ht"] = np.array(
                    [x.sum() for x in self.full_data_array[var]]
                )
                self.full_data_array["truth_ptrel"] = np.array(
                    [x / x.sum() for x in self.full_data_array[var]], dtype=object
                )

        for i, var in enumerate(self.truth_variables):
            if var == "truth_pt":
                self.truth_variables[i] = "truth_ptrel"

        self.full_data_array["truth_met_x"] = np.zeros(self.nevents, dtype=np.float32)
        self.full_data_array["truth_met_y"] = np.zeros(self.nevents, dtype=np.float32)

        for i in range(self.nevents):
            self.full_data_array["truth_met_x"][i] = (
                self.full_data_array["truth_pt"][i]
                * np.cos(self.full_data_array["truth_phi"][i])
            ).sum()
            self.full_data_array["truth_met_y"][i] = (
                self.full_data_array["truth_pt"][i]
                * np.sin(self.full_data_array["truth_phi"][i])
            ).sum()

        self.full_data_array.pop("truth_pt")

    def _load_pflow(self):
        self.pflow_variables = [el for el in self.config["pflow_variables"]]

        self.n_pflow_particles = self.tree["npflow"].array(
            library="np",
            entry_stop=self.nevents + self.entry_start,
            entry_start=self.entry_start,
        )
        if self.n_particle_mask is None:
            self.n_particle_mask = (self.n_pflow_particles < self.max_particles) & (
                self.n_pflow_particles > 0
            )
        else:
            self.n_particle_mask = (
                self.n_particle_mask
                & (self.n_pflow_particles < self.max_particles)
                & (self.n_pflow_particles > 0)
            )

        for var in tqdm(self.pflow_variables):
            self.full_data_array[var] = self.tree[var].array(
                library="np",
                entry_stop=self.nevents + self.entry_start,
                entry_start=self.entry_start,
            )
            if var == "pflow_pt":
                self.full_data_array["pflow_ht"] = np.array(
                    [x.sum() for x in self.full_data_array[var]]
                )
                self.full_data_array["pflow_ptrel"] = np.array(
                    [x / x.sum() for x in self.full_data_array[var]], dtype=object
                )
        for i, var in enumerate(self.pflow_variables):
            if var == "pflow_pt":
                self.pflow_variables[i] = "pflow_ptrel"

        self.full_data_array["pflow_met_x"] = np.zeros(self.nevents, dtype=np.float32)
        self.full_data_array["pflow_met_y"] = np.zeros(self.nevents, dtype=np.float32)

        for i in range(self.nevents):
            self.full_data_array["pflow_met_x"][i] = (
                self.full_data_array["pflow_pt"][i]
                * np.cos(self.full_data_array["pflow_phi"][i])
            ).sum()
            self.full_data_array["pflow_met_y"][i] = (
                self.full_data_array["pflow_pt"][i]
                * np.sin(self.full_data_array["pflow_phi"][i])
            ).sum()

        self.full_data_array.pop("pflow_pt")

    def _get_truth_data(self, idx):
        """
        Returns the truth particle data for the given index

        Args:
            idx (int): index of the event

        Returns:
            truth_data (dict): dictionary containing the truth data
            truth_mask (torch.Tensor): mask for the truth data
            truth_vars_scales (dict): dictionary containing the shift and scale for the truth data
            global_data (torch.Tensor): tensor containing the global data (shift and scale from the truth data)
        """
        n_truth_particles = self.n_truth_particles[idx]
        truth_start, truth_end = self.truth_cumsum[idx], self.truth_cumsum[idx + 1]

        truth_vars = {
            key.replace("truth_", ""): self.full_data_array[key][truth_start:truth_end]
            for key in self.truth_variables
        }
        truth_vars["class"] = truth_vars["class"]

        truth_vars_scales = {}
        truth_data = {}
        global_data = {}

        truth_idx = torch.argsort(truth_vars["ptrel"], descending=True)

        for key in self.truth_variables:
            var_name = key.replace("truth_", "")
            if var_name == "class":
                truth_data[var_name] = F.one_hot(
                    truth_vars[var_name][truth_idx], 5
                ).float()
                continue
            var_transform = self.var_transform_dict[var_name]
            shift, scale = self.truth_vars_shift_scales[key][idx]
            global_data[var_name + "_shift"] = shift
            global_data[var_name + "_scale"] = scale
            if var_transform.do_pes:
                truth_vars_scales[var_name] = shift, scale
            else:
                shift, scale = None, None
            truth_data[var_name] = (
                var_transform.transform(truth_vars[var_name][truth_idx], shift, scale)
                .float()
                .unsqueeze(-1)
            )

        truth_mask = torch.zeros(self.max_particles)
        truth_mask[:n_truth_particles] = 1

        return truth_data, truth_mask, truth_vars_scales, global_data

    def _get_pflow_data(self, idx, truth_vars_scales):
        """
        Returns the pflow particle data for the given index

        Args:
            idx (int): index of the event
            truth_vars_scales (dict): dictionary containing the shift and scale for the truth data

        Returns:
            pflow_data (dict): dictionary containing the pflow data
            pflow_mask (torch.Tensor): mask for the pflow data
        """
        n_pflow_particles = self.n_pflow_particles[idx]

        pflow_start, pflow_end = self.pflow_cumsum[idx], self.pflow_cumsum[idx + 1]

        pflow_vars = {
            key.replace("pflow_", ""): self.full_data_array[key][pflow_start:pflow_end]
            for key in self.pflow_variables
        }
        pflow_data = {}

        pflow_idx = torch.argsort(pflow_vars["ptrel"], descending=True)

        for key in self.pflow_variables:
            var_name = key.replace("pflow_", "")
            if var_name == "class":
                pflow_data[var_name] = F.one_hot(
                    pflow_vars[var_name][pflow_idx].long(), 5
                ).float()
                continue
            shift, scale = truth_vars_scales.get(var_name, (None, None))
            pflow_data[var_name] = (
                self.var_transform_dict[var_name]
                .transform(pflow_vars[var_name][pflow_idx], shift, scale)
                .float()
                .unsqueeze(-1)
            )

        pflow_mask = torch.zeros(self.max_particles)
        pflow_mask[:n_pflow_particles] = 1

        return pflow_data, pflow_mask

    def get_particle_data(self, idx):
        truth_data, truth_mask, truth_vars_scales, global_data = self._get_truth_data(
            idx
        )
        truth_data = torch.cat(
            [truth_data[key.replace("truth_", "")] for key in self.truth_variables], -1
        )
        truth_data = do_padding(truth_data, self.max_particles)

        pflow_data, pflow_mask = self._get_pflow_data(idx, truth_vars_scales)
        pflow_data = torch.cat(
            [pflow_data[key.replace("pflow_", "")] for key in self.pflow_variables],
            -1,
        )
        pflow_data = do_padding(pflow_data, self.max_particles)
        mask = torch.stack([truth_mask, pflow_mask], -1)
        # mask = mask.bool()
        mask = mask.to(torch.float32)

        global_data = torch.stack(list(global_data.values()), -1).to(torch.float32)

        global_data = torch.cat(
            [
                global_data if self.use_scale_info else torch.FloatTensor([]),
                self.scaled_global_data[idx],
            ],
            -1,
        )

        if self.sin_cos:
            truth_data_sin = torch.sin(truth_data[..., 2] * 1.814).unsqueeze(-1)
            truth_data_cos = torch.cos(truth_data[..., 2] * 1.814).unsqueeze(-1)
            truth_data = torch.cat(
                [
                    truth_data[..., :2],
                    truth_data_sin,
                    truth_data_cos,
                    truth_data[..., 3:],
                ],
                -1,
            )
            pflow_data_sin = torch.sin(pflow_data[..., 2] * 1.814).unsqueeze(-1)
            pflow_data_cos = torch.cos(pflow_data[..., 2] * 1.814).unsqueeze(-1)
            pflow_data = torch.cat(
                [
                    pflow_data[..., :2],
                    pflow_data_sin,
                    pflow_data_cos,
                    pflow_data[..., 3:],
                ],
                -1,
            )

        return truth_data, pflow_data, mask, global_data

    def get_event_data(self, idx):
        n_truth_particles = self.n_truth_particles[idx]
        n_pflow_particles = self.n_pflow_particles[idx]

        truth_data, truth_mask, _, global_data = self._get_truth_data(idx)
        truth_data = torch.cat(
            [truth_data[key.replace("truth_", "")] for key in self.truth_variables], -1
        )
        truth_data = do_padding(truth_data, self.max_particles)

        pflow_data = torch.zeros(4)
        pflow_data[0] = (
            self.var_transform_dict["ht"]
            .transform(self.full_data_array["pflow_ht"][idx])
            .float()
        )
        pflow_data[1] = (
            self.var_transform_dict["npart"]
            .transform(torch.tensor(n_pflow_particles))
            .float()
        )
        pflow_data[2] = (
            self.var_transform_dict["met_x"]
            .transform(self.full_data_array["pflow_met_x"][idx].float())
            .float()
        )
        pflow_data[3] = (
            self.var_transform_dict["met_y"]
            .transform(self.full_data_array["pflow_met_y"][idx].float())
            .float()
        )
        pflow_data = pflow_data.to(torch.float32)

        global_data = torch.stack(list(global_data.values()), -1).to(torch.float32)
        global_data = torch.cat(
            [
                global_data if self.use_scale_info else torch.FloatTensor([]),
                self.scaled_global_data[idx],
            ],
            -1,
        )
        truth_mask = truth_mask

        if n_truth_particles == 0:
            truth_data = torch.zeros_like(truth_data).float()
            pflow_data = torch.zeros_like(pflow_data).float()
            truth_mask = torch.zeros(self.max_particles, dtype=bool)
            global_data = torch.zeros_like(global_data).float()

        truth_mask = truth_mask.to(torch.float32)
        return truth_data, pflow_data, truth_mask, global_data

    def get_eval_data(self, idx):
        n_truth_particles = self.n_truth_particles[idx]

        truth_data, truth_mask, _, global_data = self._get_truth_data(idx)
        truth_data = torch.cat(
            [truth_data[key.replace("truth_", "")] for key in self.truth_variables], -1
        )
        truth_data = do_padding(truth_data, self.max_particles)

        global_data = torch.stack(list(global_data.values()), -1).to(torch.float32)

        global_data = torch.cat(
            [
                global_data if self.use_scale_info else torch.FloatTensor([]),
                self.var_transform_dict["met_x"].transform(
                    self.full_data_array["truth_met_x"][idx].float().unsqueeze(-1)
                ),
                self.var_transform_dict["met_y"].transform(
                    self.full_data_array["truth_met_y"][idx].float().unsqueeze(-1)
                ),
                self.var_transform_dict["npart"]
                .transform(torch.tensor(n_truth_particles).float().unsqueeze(-1))
                .float(),
                self.var_transform_dict["ht"]
                .transform(self.full_data_array["truth_ht"][idx].float().unsqueeze(-1))
                .float(),
            ],
            -1,
        )

        return truth_data, truth_mask, global_data

    def __len__(self):
        return len(self.n_truth_particles)

    # def __getitem__(self, idx):
    #     if self.mode == "eval":
    #         return self.get_eval_data(idx)
    #     if self.train_type == "particle":
    #         return self.get_particle_data(idx)
    #     elif self.train_type == "evt":
    #         return self.get_event_data(idx)
    def __getitem__(self, idx):
        if self.mode == "eval":
            item = self.get_eval_data(idx)
        elif self.train_type == "particle":
            item = self.get_particle_data(idx)
        elif self.train_type == "evt":
            item = self.get_event_data(idx)
        else:
            raise ValueError(f"Unknown train_type: {self.train_type}")

        self.type_check_debug(item, idx)
        return item

    def type_check_debug(self, item, idx):
        if isinstance(item, tuple):
            iterator = enumerate(item)
        elif isinstance(item, dict):
            iterator = item.items()
        else:
            print(f"⚠️ Skipping check: unexpected type at idx {idx}: {type(item)}")
            return

        for k, v in iterator:
            if isinstance(v, torch.Tensor) and v.dtype == torch.bool:
                print(f"\n❌ Boolean tensor at idx {idx}")
                print(f"Key: {k}, Shape: {v.shape}, Sample values: {v.flatten()[:10]}")
                torch.save({"idx": idx, "key": k, "data": v, "full_item": item}, f"bad_event_{idx}.pt")
                raise TypeError(f"Boolean tensor in batch (key: {k}) at idx {idx}")
