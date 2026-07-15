import numpy as np
import torch
import torch.nn.functional as F
import uproot
from torch.utils.data import Dataset

from utils.datasetloader import VarTransform, do_padding, normalize


class FastSimStreamingDataset(Dataset):
    """Dataset that streams one event at a time from disk."""

    def __init__(
        self,
        filename,
        config=None,
        reduce_ds=1.0,
        entry_start=0,
        mode="train",
    ):
        super().__init__()
        self.filename = filename
        self.config = config or {}
        self.mode = mode
        self.max_particles = self.config["max_particles"]
        self.train_type = self.config.get("train_type", "particle")
        self.use_scale_info = self.config.get("use_scale_info", True)
        self.zero_neutral_vtx = self.config.get("zero_neutral_vtx", False)
        self.sin_cos = self.config.get("sin_cos", False)

        self.truth_branches = list(self.config["truth_variables"])
        self.pflow_branches = list(self.config.get("pflow_variables", []))
        self.entry_start = entry_start

        self.truth_features = list(self.truth_branches)
        for idx, branch in enumerate(self.truth_features):
            if branch == "truth_pt":
                self.truth_features[idx] = "truth_ptrel"
        self.truth_feature_names = [
            branch.replace("truth_", "") for branch in self.truth_features
        ]

        self.pflow_features = list(self.pflow_branches)
        for idx, branch in enumerate(self.pflow_features):
            if branch == "pflow_pt":
                self.pflow_features[idx] = "pflow_ptrel"
        self.pflow_feature_names = [
            branch.replace("pflow_", "") for branch in self.pflow_features
        ]

        self.var_transform_dict = {
            key: VarTransform(key, val)
            for key, val in self.config["var_transform"].items()
        }

        self.indices = self._select_indices(reduce_ds)
        self._branches_needed = sorted(
            set(self.truth_branches)
            | ({"ntruth"} if "ntruth" not in self.truth_branches else set())
            | (set(self.pflow_branches) if self.mode == "train" else set())
            | ({"npflow"} if self.mode == "train" and "npflow" not in self.pflow_branches else set())
        )

        self._tree = None
        self._file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        entry = int(self.indices[idx])
        arrays = self._read_event(entry)

        if self.mode == "eval":
            return self._get_eval_sample(arrays)

        if self.train_type == "evt":
            return self._get_event_sample(arrays)

        return self._get_particle_sample(arrays)

    def __del__(self):
        if self._file is not None:
            self._file.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tree"] = None
        state["_file"] = None
        return state

    def _select_indices(self, reduce_ds):
        with uproot.open(self.filename) as root_file:
            tree = root_file["evt_tree"]
            total = tree.num_entries
            remaining = total - self.entry_start
            if remaining <= 0:
                raise ValueError("entry_start beyond available events")
            if isinstance(reduce_ds, float) and 0 < reduce_ds < 1:
                span = int(remaining * reduce_ds)
            elif reduce_ds:
                span = int(reduce_ds)
            else:
                span = remaining
            span = min(span, remaining)
            stop = self.entry_start + span

            branches = ["ntruth"]
            if self.mode == "train":
                branches.append("npflow")
            arrays = tree.arrays(
                branches,
                entry_start=self.entry_start,
                entry_stop=stop,
                library="np",
            )
            truth_mask = (arrays["ntruth"] > 0) & (arrays["ntruth"] < self.max_particles)
            if self.mode == "train":
                pf_mask = (arrays["npflow"] > 0) & (arrays["npflow"] < self.max_particles)
                valid = truth_mask & pf_mask
            else:
                valid = truth_mask
            valid_idx = np.nonzero(valid)[0]
            removed = span - len(valid_idx)
            if removed:
                print(f"[StreamingDataset] dropped {removed} invalid events")
            return self.entry_start + valid_idx.astype(np.int64)

    def _ensure_tree(self):
        if self._tree is None:
            self._file = uproot.open(self.filename)
            self._tree = self._file["evt_tree"]

    def _read_event(self, entry):
        self._ensure_tree()
        data = self._tree.arrays(
            self._branches_needed,
            entry_start=entry,
            entry_stop=entry + 1,
            library="np",
        )
        return {k: v[0] for k, v in data.items()}

    def _prepare_truth(self, arrays):
        truth = {}
        for branch in self.truth_branches:
            vals = arrays[branch]
            name = branch.replace("truth_", "")
            dtype = torch.long if name == "class" else torch.float32
            tensor = torch.tensor(vals, dtype=dtype)
            if name == "eta":
                tensor = torch.clamp(tensor, -3, 3)
            elif name == "phi":
                tensor = normalize(tensor)
            truth[name] = tensor

        pt = truth["pt"]
        phi = truth["phi"]
        n_truth = pt.numel()
        ht = pt.sum()
        safe_ht = ht if ht > 0 else torch.tensor(1.0)
        truth["ptrel"] = pt / safe_ht
        met_x = (pt * torch.cos(phi)).sum()
        met_y = (pt * torch.sin(phi)).sum()

        ordering = torch.argsort(truth["ptrel"], descending=True)

        truth_data = {}
        truth_scales = {}
        scale_values = []
        for feature, name in zip(self.truth_features, self.truth_feature_names):
            values = truth[name]
            if name == "class":
                truth_data[name] = F.one_hot(values[ordering], 5).float()
                continue
            transform = self.var_transform_dict[name]
            shift, scale = transform.calculate(values)
            if self.use_scale_info:
                scale_values.append(torch.tensor([shift, scale]))
            if transform.do_pes:
                truth_scales[name] = (shift, scale)
                sh, sc = shift, scale
            else:
                sh = sc = None
            normed = transform.transform(values[ordering], sh, sc)
            truth_data[name] = normed.float().unsqueeze(-1)

        truth_tensor = torch.cat(
            [truth_data[name] for name in self.truth_feature_names], dim=-1
        )
        truth_tensor = do_padding(truth_tensor, self.max_particles)
        truth_mask = torch.zeros(self.max_particles)
        truth_mask[:n_truth] = 1

        scale_tensor = (
            torch.stack(scale_values) if scale_values else torch.empty(0)
        )

        summary = {
            "ht": ht.detach().clone(),
            "met_x": met_x.detach().clone(),
            "met_y": met_y.detach().clone(),
            "npart": torch.tensor(float(n_truth)),
        }
        return truth_tensor, truth_mask.bool(), truth_scales, scale_tensor, summary

    def _prepare_pflow(self, arrays, truth_scales):
        pf = {}
        for branch in self.pflow_branches:
            vals = arrays[branch]
            name = branch.replace("pflow_", "")
            dtype = torch.long if name == "class" else torch.float32
            tensor = torch.tensor(vals, dtype=dtype)
            if name == "eta":
                tensor = torch.clamp(tensor, -3, 3)
            elif name == "phi":
                tensor = normalize(tensor)
            pf[name] = tensor

        pt = pf["pt"]
        phi = pf["phi"]
        n_pf = pt.numel()
        ht = pt.sum()
        safe_ht = ht if ht > 0 else torch.tensor(1.0)
        pf["ptrel"] = pt / safe_ht
        met_x = (pt * torch.cos(phi)).sum()
        met_y = (pt * torch.sin(phi)).sum()

        if self.zero_neutral_vtx:
            neutral_mask = pf["class"] > 2
            for coord in ["vx", "vy", "vz"]:
                if coord in pf:
                    temp = pf[coord].clone()
                    temp[neutral_mask] = 0
                    pf[coord] = temp

        ordering = torch.argsort(pf["ptrel"], descending=True)

        pf_data = {}
        for feature, name in zip(self.pflow_features, self.pflow_feature_names):
            values = pf[name]
            if name == "class":
                pf_data[name] = F.one_hot(values[ordering].long(), 5).float()
                continue
            shift, scale = truth_scales.get(name, (None, None))
            normed = self.var_transform_dict[name].transform(
                values[ordering], shift, scale
            )
            pf_data[name] = normed.float().unsqueeze(-1)

        pf_tensor = torch.cat(
            [pf_data[name] for name in self.pflow_feature_names], dim=-1
        )
        pf_tensor = do_padding(pf_tensor, self.max_particles)
        pf_mask = torch.zeros(self.max_particles)
        pf_mask[:n_pf] = 1

        summary = {
            "ht": ht.detach().clone(),
            "met_x": met_x.detach().clone(),
            "met_y": met_y.detach().clone(),
            "npart": torch.tensor(float(n_pf)),
        }
        return pf_tensor, pf_mask.bool(), summary

    def _stack_global(self, truth_summary, pf_summary=None):
        feats = [
            self.var_transform_dict["met_x"].transform(
                truth_summary["met_x"].unsqueeze(0)
            ).squeeze(0),
            self.var_transform_dict["met_y"].transform(
                truth_summary["met_y"].unsqueeze(0)
            ).squeeze(0),
            self.var_transform_dict["npart"].transform(
                truth_summary["npart"].unsqueeze(0)
            ).squeeze(0),
            self.var_transform_dict["ht"].transform(
                truth_summary["ht"].unsqueeze(0)
            ).squeeze(0),
        ]
        if pf_summary is not None:
            feats.extend(
                [
                    self.var_transform_dict["met_x"].transform(
                        pf_summary["met_x"].unsqueeze(0)
                    ).squeeze(0),
                    self.var_transform_dict["met_y"].transform(
                        pf_summary["met_y"].unsqueeze(0)
                    ).squeeze(0),
                    self.var_transform_dict["npart"].transform(
                        pf_summary["npart"].unsqueeze(0)
                    ).squeeze(0),
                    self.var_transform_dict["ht"].transform(
                        pf_summary["ht"].unsqueeze(0)
                    ).squeeze(0),
                ]
            )
        return torch.stack(feats)

    def _apply_sin_cos(self, tensor):
        phi = tensor[..., 2]
        sin = torch.sin(phi * 1.814).unsqueeze(-1)
        cos = torch.cos(phi * 1.814).unsqueeze(-1)
        return torch.cat([tensor[..., :2], sin, cos, tensor[..., 3:]], dim=-1)

    def _get_particle_sample(self, arrays):
        truth_tensor, truth_mask, truth_scales, scale_tensor, truth_summary = (
            self._prepare_truth(arrays)
        )
        pf_tensor, pf_mask, pf_summary = self._prepare_pflow(arrays, truth_scales)

        if self.sin_cos:
            truth_tensor = self._apply_sin_cos(truth_tensor)
            pf_tensor = self._apply_sin_cos(pf_tensor)

        mask = torch.stack([truth_mask, pf_mask], dim=-1).bool()
        global_stack = self._stack_global(truth_summary, pf_summary)
        if self.use_scale_info and scale_tensor.numel():
            global_data = torch.cat([scale_tensor.flatten(), global_stack])
        else:
            global_data = global_stack

        return (
            truth_tensor.float(),
            pf_tensor.float(),
            mask,
            global_data.float(),
        )

    def _get_event_sample(self, arrays):
        truth_tensor, truth_mask, truth_scales, scale_tensor, truth_summary = (
            self._prepare_truth(arrays)
        )
        pf_tensor = torch.zeros(4)
        pf_summary = None
        if self.mode == "train":
            _, _, pf_summary = self._prepare_pflow(arrays, truth_scales)
            pf_tensor[0] = self.var_transform_dict["ht"].transform(
                pf_summary["ht"].unsqueeze(0)
            )
            pf_tensor[1] = self.var_transform_dict["npart"].transform(
                pf_summary["npart"].unsqueeze(0)
            )
            pf_tensor[2] = self.var_transform_dict["met_x"].transform(
                pf_summary["met_x"].unsqueeze(0)
            )
            pf_tensor[3] = self.var_transform_dict["met_y"].transform(
                pf_summary["met_y"].unsqueeze(0)
            )
        global_stack = self._stack_global(truth_summary, pf_summary)
        if self.use_scale_info and scale_tensor.numel():
            global_data = torch.cat([scale_tensor.flatten(), global_stack])
        else:
            global_data = global_stack

        return (
            truth_tensor.float(),
            pf_tensor.float(),
            truth_mask.bool(),
            global_data.float(),
        )

    def _get_eval_sample(self, arrays):
        truth_tensor, truth_mask, _, scale_tensor, truth_summary = self._prepare_truth(
            arrays
        )
        global_stack = self._stack_global(truth_summary)
        if self.use_scale_info and scale_tensor.numel():
            global_data = torch.cat([scale_tensor.flatten(), global_stack])
        else:
            global_data = global_stack
        return truth_tensor.float(), truth_mask.bool(), global_data.float()
