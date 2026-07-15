"""Lightning module for unpaired ALEPH→DELPHI event-level domain translation.

Predicts DELPHI event properties (Ht, n_reco, MET_x, MET_y) from ALEPH reco
particles using FlowNumPFNet + TargetConditionalFlowMatcher.  The event model
is run first at inference time; its output drives the particle count passed to
the particle model (DomainTranslationLightning.sample).

Mini-batch OT pairing on Ht aligns unpaired ALEPH/DELPHI events within each
training batch — identical strategy to the particle model.
"""

import sys

import torch
import torch.nn.functional as F
from pytorch_lightning.core.module import LightningModule
from torch.utils.data import DataLoader

sys.path.append("./models/")

from models.flow_npf_model import FlowNumPFNet
from models.sampler import pndm_sampler
from utils.conditional_flow_matching import TargetConditionalFlowMatcher
from utils.custom_scheduler import CosineAnnealingWarmupRestarts
from utils.datasetloader import VarTransform
from utils.domain_datasetloader import DomainRecoDataset
from utils.lion_opt import Lion
from utils.minibatch_ot import sinkhorn_ot_permutation, sorted_ot_permutations


class DomainEvtLightning(LightningModule):
    """Event-level ALEPH→DELPHI domain translation via flow matching.

    Predicts noisy_dim=4 DELPHI event features:
      [Ht_delphi_norm, n_delphi_norm, MET_x_delphi_norm, MET_y_delphi_norm]
    conditioned on ALEPH reco particles (truth_data) and ALEPH event globals.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_steps = config["n_steps"]
        self.opt = config.get("opt", "adamw")
        self.ot_mode = config.get("ot_mode", "sorted_ot")
        self.noisy_dim = config.get("noisy_dim", 4)
        self.sigma = config.get("sigma", 0.0001)

        self.net = FlowNumPFNet(config, noisy_dim=self.noisy_dim)
        self.FM = TargetConditionalFlowMatcher(sigma=self.sigma)
        self.loss = torch.nn.MSELoss()

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
        """Pair source/target batches via mini-batch OT.

        sorted_ot  — baseline: sort both batches by Ht, pair by rank (1D OT)
        sinkhorn   — upgrade: weighted multi-observable cost + Sinkhorn + Hungarian
        """
        src_reco, src_mask, src_global = src_batch
        _, _, tgt_global = tgt_batch

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
            tgt_global[pt],
        )

    # ------------------------------------------------------------------
    # Forward (shared between train and val)
    # ------------------------------------------------------------------

    def _forward(self, src_batch, tgt_batch):
        src_reco, src_mask, src_global, tgt_global = self._pair_batch(
            src_batch, tgt_batch
        )

        x0 = torch.randn_like(tgt_global)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, tgt_global)

        vt = self.net(xt, src_reco, src_mask, timestep=t, global_data=src_global)

        ht_loss = self.loss(vt[..., 0], ut[..., 0])
        npf_loss = self.loss(vt[..., 1], ut[..., 1])
        met_x_loss = self.loss(vt[..., 2], ut[..., 2])
        met_y_loss = self.loss(vt[..., 3], ut[..., 3])

        total_loss = (ht_loss + npf_loss + met_x_loss + met_y_loss) / 4
        return total_loss, {
            "ht_loss": ht_loss,
            "npf_loss": npf_loss,
            "met_x_loss": met_x_loss,
            "met_y_loss": met_y_loss,
        }

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        total_loss, losses = self._forward(batch["source"], batch["target"])
        losses["total_loss"] = total_loss
        self.log_dict(
            {f"train/{k}": v.item() for k, v in losses.items()},
            batch_size=batch["source"][0].shape[0],
            sync_dist=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, losses = self._forward(batch["source"], batch["target"])
        losses["total_loss"] = total_loss
        self.log_dict(
            {f"val/{k}": v.item() for k, v in losses.items()},
            batch_size=batch["source"][0].shape[0],
            sync_dist=True,
        )
        self.log("val_loss_avg", total_loss, batch_size=batch["source"][0].shape[0], sync_dist=True)
        return total_loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, src_reco, src_mask, src_global, n_steps=None):
        """Predict DELPHI event features from ALEPH reco particles.

        Args:
            src_reco:   [bs, max_p, feature_dim]  ALEPH reco features (encoded)
            src_mask:   [bs, max_p]  bool
            src_global: [bs, global_dim]
            n_steps:    int, optional

        Returns:
            [bs, noisy_dim] predicted DELPHI event features (normalised)
              dim 0: Ht_norm, dim 1: n_norm, dim 2: MET_x_norm, dim 3: MET_y_norm
        """
        if n_steps is None:
            n_steps = self.n_steps
        pflow_shape = (src_reco.shape[0], self.noisy_dim)
        return pndm_sampler(
            self.net,
            src_reco,
            pflow_shape,
            src_mask,
            src_global,
            n_steps=n_steps,
            dt=0.0,
            save_seq=False,
            zero_init_padded=False,
            reverse_time=False,
        )[0]

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
