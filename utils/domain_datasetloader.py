"""Dataset for unpaired reco-to-reco domain translation.

Loads reconstructed particles from a single ROOT file (ALEPH or DELPHI).
Unlike FastSimDataset, no truth particles are loaded — only pflow_* branches.

Returns per-event tuples:
  reco_data  : [max_particles, feature_dim]  normalised, encoded particle features
  reco_mask  : [max_particles]  bool          True for real particles
  global_data: [4]  float                     [Ht_norm, n_norm, MET_x_norm, MET_y_norm]

Feature encoding with sin_cos=True and variables=[ptrel, eta, phi, class]:
  [ptrel_norm, eta_norm, sin(phi*1.814), cos(phi*1.814),
   class_0, class_1, class_2, class_3, class_4]
  total = 4 + 5 = 9 dims  (matches FlowNet with truth_in_dim = len(variables)+5)
"""

import gc

import numpy as np
import torch
import torch.nn.functional as F
import uproot
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.datasetloader import VarTransform, do_padding, normalize


class DomainRecoDataset(Dataset):
    """Loads reco particles from one ROOT file for domain translation."""

    def __init__(self, filename, config, reduce_ds=None, entry_start=0):
        super().__init__()
        self.config = config
        self.max_particles = config["max_particles"]
        self.sin_cos = config.get("sin_cos", True)

        self.var_transform_dict = {
            key: VarTransform(key, val)
            for key, val in config["var_transform"].items()
        }

        self._load(filename, reduce_ds, entry_start)

    # ------------------------------------------------------------------
    def _load(self, filename, reduce_ds, entry_start):
        print(f"[DomainRecoDataset] Loading {filename}", flush=True)
        f = uproot.open(filename, num_workers=6)
        tree = f["evt_tree"]

        nevents_raw = tree.num_entries - entry_start
        if reduce_ds is None or reduce_ds <= 0:
            nevents = nevents_raw
        elif 0 < reduce_ds < 1.0:
            nevents = int(nevents_raw * reduce_ds)
        else:
            nevents = int(reduce_ds)
        nevents = min(nevents, nevents_raw)
        entry_stop = entry_start + nevents

        branches = ["npflow", "pflow_pt", "pflow_eta", "pflow_phi", "pflow_class"]
        raw = tree.arrays(
            branches,
            library="np",
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        f.close()

        npflow = raw["npflow"].astype(np.int32)
        pflow_pt = raw["pflow_pt"]       # object array of per-event np arrays
        pflow_eta = raw["pflow_eta"]
        pflow_phi = raw["pflow_phi"]
        pflow_class = raw["pflow_class"]

        # Event-level filter
        evt_mask = (npflow > 0) & (npflow < self.max_particles)
        n_removed = (~evt_mask).sum()
        npflow = npflow[evt_mask]
        pflow_pt = pflow_pt[evt_mask]
        pflow_eta = pflow_eta[evt_mask]
        pflow_phi = pflow_phi[evt_mask]
        pflow_class = pflow_class[evt_mask]

        print(
            f"[DomainRecoDataset] {len(npflow)} valid events "
            f"(removed {n_removed} with 0 or ≥{self.max_particles} particles)",
            flush=True,
        )

        # Compute event-level scalars
        ht = np.array([float(pt.sum()) for pt in pflow_pt], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            ptrel_list = [
                pt / pt.sum() if pt.sum() > 0 else np.zeros_like(pt)
                for pt in pflow_pt
            ]
        met_x = np.array(
            [float((pt * np.cos(phi)).sum())
             for pt, phi in zip(pflow_pt, pflow_phi)],
            dtype=np.float32,
        )
        met_y = np.array(
            [float((pt * np.sin(phi)).sum())
             for pt, phi in zip(pflow_pt, pflow_phi)],
            dtype=np.float32,
        )

        # Flatten per-particle arrays
        self.pflow_ptrel = torch.tensor(
            np.concatenate(ptrel_list).astype(np.float32), dtype=torch.float32
        )
        self.pflow_eta = torch.tensor(
            np.concatenate(pflow_eta).astype(np.float32), dtype=torch.float32
        )
        self.pflow_phi = torch.tensor(
            np.concatenate(pflow_phi).astype(np.float32), dtype=torch.float32
        )
        # Class: clamp to [0, 4] (neutrinos and residuals map to 0)
        cls_flat = np.concatenate(pflow_class).astype(np.int32)
        cls_flat = np.clip(cls_flat, 0, 4)
        self.pflow_class = torch.tensor(cls_flat, dtype=torch.long)

        # Normalise eta and phi in-place
        self.pflow_eta = torch.clamp(self.pflow_eta, -3.0, 3.0)
        self.pflow_phi = normalize(self.pflow_phi)

        self.n_particles = torch.tensor(npflow, dtype=torch.long)
        self.ht = torch.tensor(ht, dtype=torch.float32)
        self.met_x = torch.tensor(met_x, dtype=torch.float32)
        self.met_y = torch.tensor(met_y, dtype=torch.float32)
        self.cumsum = np.cumsum(np.concatenate([[0], npflow]))

        # Precompute normalised global features: [Ht, n_reco, MET_x, MET_y]
        met_clamp = self.config.get("met_clamp", 1000.0)
        self.scaled_global = torch.stack(
            [
                self.var_transform_dict["ht"].transform(self.ht),
                self.var_transform_dict["npart"].transform(self.n_particles.float()),
                self.var_transform_dict["met_x"].transform(
                    torch.clamp(self.met_x, -met_clamp, met_clamp)
                ),
                self.var_transform_dict["met_y"].transform(
                    torch.clamp(self.met_y, -met_clamp, met_clamp)
                ),
            ],
            dim=-1,
        ).float()

        self.nevents = len(self.n_particles)
        gc.collect()
        print(f"[DomainRecoDataset] Done. {self.nevents} events.", flush=True)

    # ------------------------------------------------------------------
    def __len__(self):
        return self.nevents

    def __getitem__(self, idx):
        n = int(self.n_particles[idx])
        start = int(self.cumsum[idx])
        end = int(self.cumsum[idx + 1])

        ptrel = self.pflow_ptrel[start:end]
        eta = self.pflow_eta[start:end]
        phi = self.pflow_phi[start:end]
        cls = self.pflow_class[start:end]

        # Sort descending by ptrel (same convention as FastSimDataset)
        sort_idx = torch.argsort(ptrel, descending=True)
        ptrel = ptrel[sort_idx]
        eta = eta[sort_idx]
        phi = phi[sort_idx]
        cls = cls[sort_idx]

        # Normalise
        ptrel_n = self.var_transform_dict["ptrel"].transform(ptrel).float().unsqueeze(-1)
        eta_n = self.var_transform_dict["eta"].transform(eta).float().unsqueeze(-1)
        cls_oh = F.one_hot(cls.clamp(0, 4), 5).float()

        if self.sin_cos:
            sin_phi = torch.sin(phi * 1.814).unsqueeze(-1)
            cos_phi = torch.cos(phi * 1.814).unsqueeze(-1)
            reco_data = torch.cat([ptrel_n, eta_n, sin_phi, cos_phi, cls_oh], dim=-1)
        else:
            phi_n = self.var_transform_dict["phi"].transform(phi).float().unsqueeze(-1)
            reco_data = torch.cat([ptrel_n, eta_n, phi_n, cls_oh], dim=-1)

        reco_data = do_padding(reco_data, self.max_particles)

        reco_mask = torch.zeros(self.max_particles, dtype=torch.bool)
        reco_mask[:n] = True

        return reco_data, reco_mask, self.scaled_global[idx]
