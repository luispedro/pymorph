import pymorph
import numpy as np

def test_erode_mixed_types():
    A  = np.array([[1,0],[0,1]])
    f = np.zeros((4,4), np.bool)
    f[2,2] = 1
    f[3,3] = 1
    assert pymorph.erode(f,A).sum() == 1

