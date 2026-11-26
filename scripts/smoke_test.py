"""Self-test without private data, pretrained weights or a GPU."""
import torch
from mcl_asf import MCLASFPaperAligned,PaperSpecificHParams,make_two_views,paper_aligned_total_loss

def main():
    torch.set_num_threads(min(torch.get_num_threads(),2))
    torch.manual_seed(3)
    x=torch.rand(2,331,16,16)
    labels=torch.tensor([0,0]);conditions=torch.tensor([0,1])
    hp=PaperSpecificHParams(0.01,1.0,0.05,0.5,0.10,0.05)
    m=MCLASFPaperAligned(bands=331,num_classes=5,feat_dim=128)
    v1,v2=make_two_views(x,hp)
    o1,o2=m(v1),m(v2)
    total,parts=paper_aligned_total_loss(o1[0],o2[0],o1[1],o2[1],o1[2],o2[2],
        labels,conditions,None,hp,lambda_mcl=0.25,temperature=0.08)
    assert torch.isfinite(total)
    assert torch.allclose(o1[2].sum(dim=1),torch.ones(2),atol=1e-5)
    total.backward()
    print('PASS: forward, ASF band-weight normalization, 2-view loss, backward')
    print('Model parameters:',sum(p.numel() for p in m.parameters()))
    print('Example logits shape:',tuple(o1[0].shape),'ASF weights:',tuple(o1[2].shape))
    print('Synthetic loss:',round(total.item(),6))

if __name__=='__main__':main()
