"""
Dev-user diagnostic: how much class information does each EMG representation carry?

Nearest-prototype (Euclidean) on the 4 DEV users only (never in any reported result), cached dev
episodes, 10-way 1/5-shot. Compares the 20 Hz envelope (env20) with EMG-FS64, alone and fused with the IMU.
Used to decide whether the FS64 representation is worth the cluster time; NOT a result.

    python compare_emg_repr_dev.py   (run from the repo root; needs both tensor dicts locally)
"""
import sys, pickle, numpy as np, torch, os
sys.path[:0]=["system","system/MAML"]
os.environ["CODE_DIR"]=os.getcwd(); os.environ["RUN_DIR"]=os.environ.get("TEMP","/tmp"); os.environ["HPO_CONFIG"]="legacy"
from episodes import episode_cache as ec
from episodes.episode_numpy import materialize
from MAML.maml_data_pipeline import reorient_tensor_dict
from types import SimpleNamespace
def load(path, emg_ch):
    d=pickle.load(open(path,"rb"))
    return d, reorient_tensor_dict(d, {"emg_in_ch":emg_ch,"imu_in_ch":72})
d20,td20=load("dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl",16)
d64,td64=load("dataset/meta-learning-sup-que-ds/segfilt_fs64_tensor_dict.pkl",112)
# sanity of the new file
x=torch.cat([td64[p][c]["emg"] for p in td64 for c in range(10)])
print("FS64 tensor", tuple(x.shape), "finite:", bool(torch.isfinite(x).all()))
dev=set(d64["feature_norm"]["computed_from_dev_pids"]); print("dev pids:",sorted(dev))
oth=torch.cat([td64[p][c]["emg"].permute(0,2,1).reshape(-1,16,7,64) .permute(0,3,1,2).reshape(-1,16,7) for p in td64 if p not in dev for c in range(10)])
print("non-dev per-feature mean:",[round(v,2) for v in oth.mean((0,1)).tolist()],"std:",[round(v,2) for v in oth.std((0,1)).tolist()])
es=ec.load("dev",None)
def acc(tdA, kind, cell):
    eps=es.select(*cell, max_per_user=40)
    per={}
    for ep in eps:
        m=materialize(ep,tdA)
        if kind=="emg": s,q=m.emg_support,m.emg_query
        elif kind=="imu": s,q=m.imu_support,m.imu_query
        else: s,q=np.concatenate([m.emg_support,m.imu_support],1),np.concatenate([m.emg_query,m.imu_query],1)
        s=s.reshape(len(s),-1); q=q.reshape(len(q),-1)
        protos=np.stack([s[m.y_support==c].mean(0) for c in range(ep.n_way)])
        pred=((q[:,None]-protos[None])**2).sum(2).argmin(1)
        per.setdefault(ep.user,[]).append((pred==m.y_query).mean())
    return np.mean([np.mean(v) for v in per.values()])
for cell in ((10,1),(10,5)):
    print(f"--- dev users, nearest-prototype (Euclid), cell {cell}, 40 eps/user")
    print(f" EMG env20 only : {acc(td20,'emg',cell):.3f}")
    print(f" EMG FS64  only : {acc(td64,'emg',cell):.3f}")
    print(f" IMU only       : {acc(td20,'imu',cell):.3f}")
    print(f" env20 + IMU    : {acc(td20,'both',cell):.3f}")
    print(f" FS64  + IMU    : {acc(td64,'both',cell):.3f}")
