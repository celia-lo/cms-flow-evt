"""Lightning module for unpaired ALEPH→DELPHI reco particle domain translation.

Architecture:
- FlowNet (cross-attention transformer) with ReverseConditionalFlowMatcher
- ALEPH reco plays the role of "truth" (source condition)
- DELPHI reco plays the role of "pflow" (generation target)
- Mini-batch OT pairing (sorted on Ht) aligns unpaired events within each batch

Training produces a model that, given ALEPH reco particles, generates
DELPHI-like reco particles via flow matching in ~n_steps DPM-Solver steps.
"""

import sys

import torch
import torch.nn.functional as F
from pytorch_lightning.core.module import LightningModule
from torch.utils.data import DataLoader

sys.path.append("./models/")

from models.dpm import DPM_Solver, NoiseScheduleFlow
from models.flow_model import FlowNet
from utils.conditional_flow_matching import ReverseConditionalFlowMatcher
from utils.custom_scheduler import CosineAnnealingWarmupRestarts
from utils.datasetloader import VarTransform
from utils.domain_datasetloader import DomainRecoDataset
from utils.lion_opt import Lion
from utils.minibatch_ot import sinkhorn_ot_permutation, sorted_ot_permutations
from fs_lightning import MSEAndDirectionLoss


class DomainTranslationLightning(LightningModule):
    """Particle-level ALEPH→DELPHI domain translation via flow matching."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_steps = config["n_steps"]
        self.opt = config.get("opt", "adamw")
        self.ot_mode = config.get("ot_mode", "sorted_ot")

        self.net = FlowNet(config)
        self.FM = ReverseConditionalFlowMatcher(sigma=config.get("sigma", 0.0001))
        self.loss_fn = MSEAndDirectionLoss()

        def model_fn(x, timestep, truth, mask, global_data):
            return (1 - timestep.view(-1, 1, 1)) * self.net(
                x, truth, mask, timestep, global_data
            ) + x

        self.dpm = DPM_Solver(model_fn=model_fn, noise_schedule=NoiseScheduleFlow())

        self.var_transform_dict = {
            key: VarTransform(key, val)
            for key, val in config["var_transform"].items()
        }

    # ------------------------------------------------------------------
    # OT pairing
    # ------------------------------------------------------------------

    def _ot_weights(self, device):
        """Return per-feature weights for Sinkhorn cost, or None for uniform."""
        w = self.config.get("ot_feature_weights", None)
        if w is None:
            return None
        return torch.tensor(w, dtype=torch.float32, device=device)

    def _pair_batch(self, src_batch, tgt_batch):
        """Pair source and target events via mini-batch OT.

        sorted_ot  — baseline: sort both batches by Ht, pair by rank (1D OT)
        sinkhorn   — upgrade: weighted multi-observable cost + Sinkhorn + Hungarian
        """
        src_reco, src_mask, src_global = src_batch
        tgt_reco, tgt_mask, tgt_global = tgt_batch

        if self.ot_mode == "sorted_ot":
            ps, pt = sorted_ot_permutations(src_global[:, 0], tgt_global[:, 0])
        elif self.ot_mode == "sinkhorn":
            ps, pt = sinkhorn_ot_permutation(
                src_global, tgt_global,
                reg=self.config.get("sinkhorn_reg", 0.05),
                n_iter=self.config.get("sinkhorn_n_iter", 50),
                weights=self._ot_weights(src_global.device),
            )
        else:
            bs = src_global.shape[0]
            ps = torch.randperm(bs, device=src_global.device)
            pt = torch.randperm(bs, device=tgt_global.device)

        return (
            src_reco[ps], src_mask[ps], src_global[ps],
            tgt_reco[pt], tgt_mask[pt],
        )

    # ------------------------------------------------------------------
    # Forward (shared between train and val)
    # ------------------------------------------------------------------

    def _forward(self, src_batch, tgt_batch):
        src_reco, src_mask, src_global, tgt_reco, tgt_mask = self._pair_batch(
            src_batch, tgt_batch
        )

        # Combined mask: dim-0 = source (ALEPH), dim-1 = target (DELPHI)
        masks = torch.stack([src_mask, tgt_mask], dim=-1)  # [bs, max_p, 2]

        x0 = torch.randn_like(tgt_reco)
        t, xt, ut, _ = self.FM.sample_location_and_conditional_flow(
            x0, tgt_reco, return_noise=True
        )

        vt = self.net(xt, src_reco, masks, timestep=t, global_data=src_global)

        pf_mask = tgt_mask.unsqueeze(-1)  # [bs, max_p, 1]
        loss = self.loss_fn(ut * pf_mask, vt * pf_mask) / pf_mask.sum() / ut.shape[-1]
        return loss

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self._forward(batch["source"], batch["target"])
        self.log("train_loss", loss, batch_size=batch["source"][0].shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward(batch["source"], batch["target"])
        self.log("val_loss_avg", loss, batch_size=batch["source"][0].shape[0], sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, src_reco, src_mask, tgt_n, src_global, n_steps=None):
        """Generate DELPHI-like particles from ALEPH reco.

        Args:
            src_reco:   [bs, max_p, feature_dim]  ALEPH reco features (encoded)
            src_mask:   [bs, max_p]  bool          ALEPH particle mask
            tgt_n:      [bs]  int                  predicted DELPHI particle count
            src_global: [bs, global_dim]           ALEPH event-level features
            n_steps:    int, optional              DPM steps (defaults to config n_steps)

        Returns:
            [bs, max_p, feature_dim] generated DELPHI-like particles (encoded space)
        """
        if n_steps is None:
            n_steps = self.n_steps

        bs, max_p = src_mask.shape
        tgt_mask = torch.zeros(bs, max_p, dtype=torch.bool, device=src_reco.device)
        for i, n in enumerate(tgt_n):
            n_clamped = int(n.clamp(1, max_p).item())
            tgt_mask[i, :n_clamped] = True

        masks = torch.stack([src_mask, tgt_mask], dim=-1)  # [bs, max_p, 2]
        fs_in_dim = self.net.fs_in_dim

        return self.dpm.sample(
            torch.randn(bs, max_p, fs_in_dim, device=src_reco.device),
            truth=src_reco,
            mask=masks,
            global_data=src_global,
            steps=n_steps,
            method="multistep",
            skip_type="time_uniform_flow",
            order=2,
        ), tgt_mask

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        if self.opt == "lion":
            optimizer = Lion(self.parameters(), lr=float(self.config["learningrate"]))
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=float(self.config["learningrate"])
            )

        if not self.config.get("lr_scheduler", False):
            return optimizer

        default_sched = {
            "first_cycle_steps": 10,
            "warmup_steps": 4,
            "max_lr": 4 * float(self.config["learningrate"]),
            "min_lr": 1e-5,
            "gamma": 0.8,
        }
        default_sched.update(self.config.get("lr_scheduler_dict", {}))
        scheduler = CosineAnnealingWarmupRestarts(optimizer, **default_sched)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _make_combined_loader(
        self, src_path, tgt_path, reduce_src, reduce_tgt,
        entry_start_src, entry_start_tgt, batch_size, shuffle, num_workers
    ):
        from lightning.pytorch.utilities import CombinedLoader

        src_ds = DomainRecoDataset(
            src_path, self.config, reduce_ds=reduce_src, entry_start=entry_start_src
        )
        tgt_ds = DomainRecoDataset(
            tgt_path, self.config, reduce_ds=reduce_tgt, entry_start=entry_start_tgt
        )

        src_loader = DataLoader(
            src_ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=True, pin_memory=False,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        tgt_loader = DataLoader(
            tgt_ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=True, pin_memory=False,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        return CombinedLoader({"source": src_loader, "target": tgt_loader}, mode="min_size")

    def train_dataloader(self):
        cfg = self.config
        return self._make_combined_loader(
            src_path=cfg["source_path_train"],
            tgt_path=cfg["target_path_train"],
            reduce_src=cfg.get("reduce_ds_source_train", None),
            reduce_tgt=cfg.get("reduce_ds_target_train", None),
            entry_start_src=cfg.get("entry_start_source_train", 0),
            entry_start_tgt=cfg.get("entry_start_target_train", 0),
            batch_size=cfg["batchsize"],
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
        )

    def val_dataloader(self):
        cfg = self.config
        return self._make_combined_loader(
            src_path=cfg.get("source_path_valid", cfg["source_path_train"]),
            tgt_path=cfg.get("target_path_valid", cfg["target_path_train"]),
            reduce_src=cfg.get("reduce_ds_source_valid", None),
            reduce_tgt=cfg.get("reduce_ds_target_valid", None),
            entry_start_src=cfg.get("entry_start_source_valid", 0),
            entry_start_tgt=cfg.get("entry_start_target_valid", 0),
            batch_size=cfg.get("val_batchsize", cfg["batchsize"]),
            shuffle=False,
            num_workers=0,
        )
