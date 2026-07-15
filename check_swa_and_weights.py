import torch

CKPT_DIR = "/pscratch/sd/y/ylo/cms_flow_part_chain_QCD_noPU_pthat2500/saved_models/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639/ckpts"
ck0 = torch.load(f"{CKPT_DIR}/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639-epoch=00-val_loss_avg=2.2112.ckpt", map_location="cpu", weights_only=False)

# Check top-level checkpoint keys — look for SWA/callbacks/averaged_model
print("=== Top-level checkpoint keys ===")
for k in ck0.keys():
    val = ck0[k]
    if isinstance(val, dict):
        print(f"  {k}: dict with {len(val)} keys -> {list(val.keys())[:5]}")
    elif isinstance(val, list):
        print(f"  {k}: list of len {len(val)}")
    else:
        print(f"  {k}: {val}")

print()

# Look specifically for any SWA or averaged model entries
callbacks = ck0.get("callbacks", {})
print("=== Callbacks state ===")
for k, v in callbacks.items():
    print(f"  {k}: {str(v)[:200]}")

print()

# Check if state_dict keys suggest this is an AveragedModel (SWA)
# AveragedModel wraps params with "module." prefix or "n_averaged" buffer
sd = ck0["state_dict"]
print("=== State dict key samples (looking for SWA pattern like 'module.' prefix) ===")
keys = list(sd.keys())
print("First 5:", keys[:5])
print("Last 5:", keys[-5:])
print("Total keys:", len(keys))
any_averaged = [k for k in keys if "n_averaged" in k or "module." in k]
print("SWA-pattern keys:", any_averaged[:10])

# Weight distribution sanity check: trained weights have non-trivial structure
# biases should NOT all be exactly 0 if actually trained
bias_keys = [k for k in keys if "bias" in k]
print()
print("=== Bias stats (trained biases drift from 0; untrained biases stay 0) ===")
for bk in bias_keys[:8]:
    t = sd[bk].float()
    print(f"  {bk}: mean={t.mean().item():.6f}  std={t.std().item():.6f}  max_abs={t.abs().max().item():.6f}")
