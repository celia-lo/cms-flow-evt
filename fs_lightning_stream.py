from torch.utils.data import DataLoader

from fs_lightning import FlowLightning
from utils.datasetloader_stream import FastSimStreamingDataset


class FlowLightningStreaming(FlowLightning):
    """FlowLightning variant that uses the streaming dataset."""

    def _build_dataset(self, path, reduce_key, entry_key, mode):
        reduce_ds = self.config.get(reduce_key, 1.0)
        entry_start = self.config.get(entry_key, 0)
        return FastSimStreamingDataset(
            path,
            self.config,
            reduce_ds=reduce_ds,
            entry_start=entry_start,
            mode=mode,
        )

    def train_dataloader(self):
        dataset = self._build_dataset(
            self.config["truth_path_train"],
            "reduce_ds_train",
            "entry_start_train",
            mode="train",
        )
        return DataLoader(
            dataset,
            num_workers=self.config["num_workers"],
            batch_size=self.config["batchsize"],
            drop_last=True,
            shuffle=True,
            pin_memory=False,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        dataset = self._build_dataset(
            self.config["truth_path_valid"],
            "reduce_ds_valid",
            "entry_start_valid",
            mode="train",
        )
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=self.config.get("val_batchsize", self.config["batchsize"]),
            drop_last=True,
            pin_memory=False,
            shuffle=False,
        )
