import unittest
import torch
import numpy as np
from mcl_asf import MCLASFPaperAligned,PaperSpecificHParams,make_two_views,paper_aligned_total_loss
from mcl_asf.data import preprocess_patch
from scripts.infer_cube import reflect_indices,get_patch

class TestMCLASF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(torch.get_num_threads(),2))

    def test_forward_and_loss(self):
        m=MCLASFPaperAligned(num_classes=5,bands=331,feat_dim=128)
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad), 544773)
        x=torch.rand(2,331,50,50)
        hp=PaperSpecificHParams(0.01,1.0,0.05,0.5,0.10,0.05)
        v1,v2=make_two_views(x,hp)
        o1,o2=m(v1),m(v2)
        self.assertEqual(tuple(o1[0].shape),(2,5))
        self.assertEqual(tuple(o1[2].shape),(2,331))
        self.assertTrue(torch.allclose(o1[2].sum(dim=1),torch.ones(2),atol=1e-5))
        y=torch.tensor([0,0]);c=torch.tensor([0,1])
        loss,_=paper_aligned_total_loss(o1[0],o2[0],o1[1],o2[1],o1[2],o2[2],y,c,None,hp)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(m.encoder.spatial.features[0].weight.grad)

    def test_patch_preprocess(self):
        a=np.random.default_rng(0).random((20,20,331),dtype=np.float32)
        x=preprocess_patch(a)
        self.assertEqual(tuple(x.shape),(331,50,50))
        self.assertTrue(torch.isfinite(x).all())

    def test_reflect(self):
        arr=np.arange(4*4*331,dtype=np.float32).reshape(4,4,331)
        patch=get_patch(arr,0,0,4)
        self.assertEqual(patch.shape,(4,4,331))
        self.assertTrue(np.array_equal(reflect_indices(np.array([-2,-1,0,1,4,5]),4),[2,1,0,1,2,1]))

if __name__=='__main__':unittest.main()
