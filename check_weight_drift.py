import torch

CKPT_DIR = "/pscratch/sd/y/ylo/cms_flow_part_chain_QCD_noPU_pthat2500/saved_models/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639/ckpts"
ck0_path = f"{CKPT_DIR}/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639-epoch=00-val_loss_avg=2.2112.ckpt"
ck13_path = f"{CKPT_DIR}/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639-epoch=13-val_loss_avg=2.2109.ckpt"

ck0 = torch.load(ck0_path, map_location="cpu", weights_only=False)
ck13 = torch.load(ck13_path, map_location="cpu", weights_only=False)

sd0 = ck0["state_dict"]
sd13 = ck13["state_dict"]

print("global_step ck0:", ck0.get("global_step"), " ck13:", ck13.get("global_step"))
print("num params:", len(sd0))

total_diff = 0.0
total_norm = 0.0
n_changed = 0
for k in sd0.keys():
    a = sd0[k].float()
    b = sd13[k].float()
    d = (a - b).abs().sum().item()
    n = a.abs().sum().item()
    total_diff += d
    total_norm += n
    if d > 1e-8:
        n_changed += 1

print("params changed (diff > 1e-8):", n_changed, "/", len(sd0))
print("total abs diff:", total_diff)
print("total abs norm:", total_norm)
print("ratio (diff/norm):", total_diff / total_norm if total_norm else float("nan"))
print("has optimizer_states:", ck0.get("optimizer_states") is not None)

# Show a few example params with their drift
print("\nSample param drifts:")
for k in list(sd0.keys())[:5] + list(sd0.keys())[-5:]:
    a = sd0[k].float()
    b = sd13[k].float()
    d = (a - b).abs().max().item()
    print(f"  {k}: shape={tuple(a.shape)} max_abs_diff={d:.6e}")
