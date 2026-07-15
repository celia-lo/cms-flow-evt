import torch

CKPT_DIR = "/pscratch/sd/y/ylo/cms_flow_part_chain_QCD_noPU_pthat2500/saved_models/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639/ckpts"
ck0_path = f"{CKPT_DIR}/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639-epoch=00-val_loss_avg=2.2112.ckpt"

ck0 = torch.load(ck0_path, map_location="cpu", weights_only=False)
sd0 = ck0["state_dict"]
opt_state = ck0["optimizer_states"][0]["state"]
param_names = list(sd0.keys())

n_full_nan = 0
n_partial_nan = 0
n_clean = 0
n_param_also_nan = 0

for idx, (pkey, pstate) in enumerate(opt_state.items()):
    name = param_names[idx] if idx < len(param_names) else f"<idx {pkey}>"
    exp_avg = pstate["exp_avg"]
    p = sd0[name]
    nan_frac = torch.isnan(exp_avg).float().mean().item()
    p_nan = torch.isnan(p).any().item()
    p_inf = torch.isinf(p).any().item()
    if nan_frac == 1.0:
        n_full_nan += 1
    elif nan_frac > 0:
        n_partial_nan += 1
        print(f"PARTIAL NaN: {name} shape={tuple(exp_avg.shape)} nan_frac={nan_frac:.4f}")
    else:
        n_clean += 1
    if p_nan or p_inf:
        n_param_also_nan += 1
        print(f"  -> param itself also NaN/Inf: {name}")

print()
print(f"exp_avg fully NaN:    {n_full_nan} / {len(opt_state)}")
print(f"exp_avg partial NaN:  {n_partial_nan} / {len(opt_state)}")
print(f"exp_avg clean:        {n_clean} / {len(opt_state)}")
print(f"params also NaN/Inf:  {n_param_also_nan} / {len(opt_state)}")

# check param_groups 'step' counters if present (lion doesn't track step, but check anyway)
pg = ck0["optimizer_states"][0]["param_groups"][0]
print()
print("param_group keys:", list(pg.keys()))
first_pstate = next(iter(opt_state.values()))
print("per-param state keys:", list(first_pstate.keys()))
