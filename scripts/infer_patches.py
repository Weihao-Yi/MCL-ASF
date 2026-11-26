"""One-vs-rest scores for `.npy` patches; no GT, ROI, or secret dataset required."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from mcl_asf.model import CLASS_NAMES
from mcl_asf.data import preprocess_patch
from mcl_asf.utils import load_checkpoint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--patches',nargs='+',required=True,help='HWC/CHW .npy patches')
    p.add_argument('--output',default='outputs/predictions.csv')
    args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,checkpoint=load_checkpoint(args.checkpoint,device)
    rows=[]
    with torch.inference_mode():
        for patchfile in args.patches:
            arr=np.load(patchfile,allow_pickle=False)
            x=preprocess_patch(arr).unsqueeze(0).to(device)
            logits,*_=model(x)
            probabilities=torch.softmax(logits,dim=1)[0].cpu().tolist()
            rows.append({'patch':str(patchfile),**{name:p for name,p in zip(CLASS_NAMES,probabilities)}})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=['patch']+list(CLASS_NAMES));wr.writeheader();wr.writerows(rows)
    print('Wrote',out)

if __name__=='__main__':main()
