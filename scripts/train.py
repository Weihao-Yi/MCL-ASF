"""Configurable MCL-ASF training using only a user-supplied indoor patch manifest."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from mcl_asf.data import load_manifest, PatchManifestDataset
from mcl_asf.model import MCLASFPaperAligned, CLASS_NAMES
from mcl_asf.augmentations import make_two_views
from mcl_asf.losses import PaperSpecificHParams, paper_aligned_total_loss
from mcl_asf.utils import set_seed, seed_worker


def parse_args():
    p=argparse.ArgumentParser(description="MCL-ASF training (indoor data only)")
    p.add_argument("--manifest",required=True,help="CSV with private patch paths (never committed)")
    p.add_argument("--config",default="configs/train.example.json")
    p.add_argument("--out",default="outputs/train")
    p.add_argument("--seeds",default="0")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--allow-cube-count-mismatch",action="store_true",help="For third-party data with a different number of indoor cubes")
    return p.parse_args()


def parse_seeds(value):
    result=set()
    for item in value.split(","):
        item=item.strip()
        if "-" in item:
            lo,hi=map(int,item.split("-",1))
            if hi<lo: raise ValueError("Bad seed range")
            result.update(range(lo,hi+1))
        else:result.add(int(item))
    return sorted(result)


def train_one_seed(seed,rows,config,out_dir):
    set_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources=Counter(r["source"] for r in rows)
    if set(sources)!={"indoor_single","indoor_mix3"}:
        raise ValueError("Both indoor_single and indoor_mix3 are required by the configured source sampler")
    sample_weights=np.asarray([
        config["source_mass_single"]/sources["indoor_single"] if r["source"]=="indoor_single"
        else config["source_mass_mix3"]/sources["indoor_mix3"] for r in rows
    ],dtype=np.float64)
    gen=torch.Generator().manual_seed(seed)
    sampler=WeightedRandomSampler(torch.as_tensor(sample_weights,dtype=torch.double),len(rows),replacement=True,generator=gen)
    ds=PatchManifestDataset(rows, output_size=int(config["input_size"]))
    loader=DataLoader(ds,batch_size=int(config["batch_size"]),sampler=sampler,
                      num_workers=int(config.get("num_workers",0)),worker_init_fn=seed_worker)
    counts=np.bincount([r["label"] for r in rows],minlength=5)
    if np.any(counts==0):
        raise ValueError(f"All five classes must be represented in training. Counts={counts.tolist()}")
    # Match sklearn.compute_class_weight(class_weight='balanced') with all 5 classes.
    weights=torch.tensor(len(rows)/(5.0*counts),dtype=torch.float32,device=device)
    hp=PaperSpecificHParams(**{k:float(config[k]) for k in (
        "sigma_n","max_shift","band_dropout_p","rho","lambda_feat","lambda_asf")})
    hp.validate()
    model=MCLASFPaperAligned(num_classes=5,bands=int(config["bands"]),feat_dim=int(config["feat_dim"])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["learning_rate"]),
                                 weight_decay=float(config["weight_decay"]))
    history=[]
    for epoch in range(1,int(config["epochs"])+1):
        model.train(); accum=Counter(); seen=0
        for x,y,conditions in loader:
            x=x.to(device);y=y.to(device);conditions=conditions.to(device)
            v1,v2=make_two_views(x,hp)
            optimizer.zero_grad(set_to_none=True)
            o1=model(v1);o2=model(v2)
            total,parts=paper_aligned_total_loss(
                o1[0],o2[0],o1[1],o2[1],o1[2],o2[2],
                labels=y,conditions=conditions,class_weights=weights,hparams=hp,
                lambda_mcl=float(config["lambda_mcl"]),temperature=float(config["temperature"]))
            total.backward();optimizer.step()
            n=int(y.numel());seen+=n
            accum["total"]+=float(total.detach().cpu())*n
            for k,v in parts.items():accum[k]+=float(v.detach().cpu())*n
        rec={"seed":seed,"epoch":epoch,**{k:accum[k]/max(seen,1) for k in
             ("total","classification","mcl","feature_consistency","asf_consistency")}}
        history.append(rec)
        print(f"seed={seed}, epoch={epoch}/{config['epochs']}, loss={rec['total']:.6f}",flush=True)
    out_dir.mkdir(parents=True,exist_ok=True)
    checkpoint={
        "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "model_config":{"num_classes":5,"bands":int(config["bands"]),"feat_dim":int(config["feat_dim"])},
        "train_config":config,"seed":seed,"class_names":list(CLASS_NAMES),
    }
    torch.save(checkpoint,out_dir/f"mcl_asf_seed_{seed:02d}.pth")
    with (out_dir/f"train_seed_{seed:02d}.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    print(f"Saved: {out_dir / ('mcl_asf_seed_%02d.pth' % seed)}")


def main():
    args=parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    if int(cfg["bands"])!=331 or int(cfg["num_classes"])!=5:
        raise ValueError("This paper implementation requires bands=331,num_classes=5")
    if not np.isclose(float(cfg["source_mass_single"])+float(cfg["source_mass_mix3"]),1.0):
        raise ValueError("Source probabilities must sum to 1.0")
    rows=load_manifest(args.manifest,check_files=True)
    cubecount={src:len({r['cube_id'] for r in rows if r['source']==src})
               for src in ('indoor_single','indoor_mix3')}
    if not args.allow_cube_count_mismatch and (
        cubecount['indoor_single'] != int(cfg['expected_single_cubes']) or
        cubecount['indoor_mix3'] != int(cfg['expected_mix3_cubes'])):
        raise ValueError("Indoor cube counts do not match configured paper protocol: " + str(cubecount))
    conditions=Counter(r["condition_name"] for r in rows)
    print('Preflight OK:',len(rows),'patches',cubecount,'cubes',dict(conditions))
    if args.preflight_only:return
    out_dir=Path(args.out)
    for seed in parse_seeds(args.seeds):train_one_seed(seed,rows,cfg,out_dir)

if __name__=='__main__':main()
