"""Generic HWC numpy hyperspectral cube -> four one-vs-rest response maps.

This deliberately does NOT apply percentile clipping, min-max score rescaling,
GT-guided postprocessing, or test-label-based threshold tuning.

A dense stride=2 evaluation on a 2048x2046 cube is computationally expensive;
for a faster example, use stride=20. Stride affects the output and must be
recorded when reporting paper results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mcl_asf.data import preprocess_patch
from mcl_asf.utils import load_checkpoint


def reflect_indices(indices,limit):
    if limit<=1:return np.zeros_like(indices,dtype=np.int64)
    periodic=2*(limit-1)
    j=np.mod(indices,periodic)
    return np.where(j<limit,j,periodic-j).astype(np.int64)


def get_patch(cube,y,x,patch_size=20):
    half=patch_size//2
    ys=reflect_indices(np.arange(y-half,y-half+patch_size),cube.shape[0])
    xs=reflect_indices(np.arange(x-half,x-half+patch_size),cube.shape[1])
    return cube[ys[:,None],xs[None,:],:]


def infer_cube(cube,model,device,stride=20,patch_size=20,batch_size=16):
    if cube.ndim!=3 or cube.shape[2]!=331:
        raise ValueError(f'Cube must be H×W×331, got {cube.shape}')
    h,w,_=cube.shape
    ys=list(range(0,h,stride));xs=list(range(0,w,stride))
    low=np.empty((4,len(ys),len(xs)),dtype=np.float32)
    model.eval()
    patches=[]; coords=[]
    def flush():
        if not coords:return
        x=torch.stack(patches,dim=0).to(device)
        with torch.inference_mode():
            prob=torch.softmax(model(x)[0],dim=1)[:,:4].cpu().numpy()
        for (i,j),p in zip(coords,prob):low[:,i,j]=p
        patches.clear();coords.clear()
    for i,y in enumerate(ys):
        for j,x in enumerate(xs):
            patches.append(preprocess_patch(np.asarray(get_patch(cube,y,x,patch_size))))
            coords.append((i,j))
            if len(coords)==batch_size:flush()
        if (i+1)%25==0:print(f'grid rows {i+1}/{len(ys)}',flush=True)
    flush()
    # Bilinear interpolation of raw model probabilities; no rank-destroying percentile clipping.
    full=F.interpolate(torch.from_numpy(low)[None],size=(h,w),mode='bilinear',align_corners=False)[0].numpy()
    return full.astype(np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cube',required=True,help='HWC .npy cube; raw data not included in repository')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='outputs/score4_raw.npy')
    p.add_argument('--stride',type=int,default=20)
    p.add_argument('--patch-size',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    if min(args.stride,args.patch_size,args.batch_size)<1:raise ValueError('All numeric parameters must be >=1')
    cube=np.load(args.cube,mmap_mode='r',allow_pickle=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,_=load_checkpoint(args.checkpoint,device)
    scores=infer_cube(cube,model,device,stride=args.stride,patch_size=args.patch_size,batch_size=args.batch_size)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.save(out,scores)
    (out.with_suffix('.json')).write_text(json.dumps({
        'source_cube_name':Path(args.cube).name,'score_shape':list(scores.shape),
        'stride':args.stride,'patch_size':args.patch_size,'batch_size':args.batch_size,
        'score_type':'interpolated raw softmax per target class; NOT calibrated',
        'class_order':['steel_ball','tungsten_bead','tungsten_block','stone'],
    },indent=2),encoding='utf-8')
    print('Saved',out,'shape',scores.shape)

if __name__=='__main__':main()
