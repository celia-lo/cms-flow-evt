import uproot
import numpy as np

path = "my_sample/Parnassus_ntuple_minbias_noPU_merge.root"
max_events = 10_000  # adjust if you want more

with uproot.open(path) as file:
    tree = file["evt_tree"]
    arrays = tree.arrays(["npflow", "ntruth"], entry_stop=max_events, library="np")
    npflow = arrays["npflow"]
    ntruth = arrays["ntruth"]

count = len(npflow)
print(f"events        : {count}")
print(f"mean npflow   : {npflow.mean():.2f}")
print(f"median npflow : {np.median(npflow):.1f}")
print(f"fraction npflow == 0 : {(npflow == 0).sum() / count:.3f}")
print(f"fraction npflow > 5  : {(npflow > 5).sum() / count:.3f}")
print(f"fraction npflow > 10 : {(npflow > 10).sum() / count:.3f}")
print(f"mean ntruth   : {ntruth.mean():.1f}")
print(f"median ntruth : {np.median(ntruth):.1f}")

