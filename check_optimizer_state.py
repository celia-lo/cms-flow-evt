import torch

CKPT_DIR = "/pscratch/sd/y/ylo/cms_flow_part_chain_QCD_noPU_pthat2500/saved_models/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639/ckpts"
ck0_path = f"{CKPT_DIR}/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639-epoch=00-val_loss_avg=2.2112.ckpt"
ck13_path = f"{CKPT_DIR}/rcfm_cms_part_QCD_noPU_pthat2500_20260607-T050639-epoch=13-val_loss_avg=2.2109.ckpt"

for label, path in [("epoch0", ck0_path), ("epoch13", ck13_path)]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(f"=== {label} (global_step={ck.get('global_step')}) ===")
    opt_states = ck.get("optimizer_states")
    print("  num optimizer_states entries:", len(opt_states) if opt_states else 0)
    if opt_states:
        st = opt_states[0]
        print("  top-level keys:", list(st.keys()))
        state = st.get("state", {})
        print("  num param states:", len(state))
        # peek at first param's state dict
        if state:
            first_key = next(iter(state))
            pstate = state[first_key]
            print(f"  first param state keys: {list(pstate.keys())}")
            for k, v in pstate.items():
                if torch.is_tensor(v):
                    print(f"    {k}: shape={tuple(v.shape)} sample={v.flatten()[:3].tolist()}")
                else:
                    print(f"    {k}: {v}")
    lr_schedulers = ck.get("lr_schedulers")
    print("  lr_schedulers:", lr_schedulers)
    print("  loops/epoch_loop state (if present):", ck.get("loops", {}).keys() if ck.get("loops") else None)
    print()
