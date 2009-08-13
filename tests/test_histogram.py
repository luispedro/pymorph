import numpy as np
import pymorph

def test_histogram():
    A = (np.random.rand(200,300)*255).astype(np.uint8)
    H = pymorph.histogram(A)
    for v,c in enumerate(H):
        assert (A == v).sum() == c
