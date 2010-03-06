import pymorph
import numpy as np
def test_smoke_skelm():
    f = np.zeros((9,9), bool)
    f[3:6, 4:7] = True
    f[4] = True
    assert pymorph.skelm(f).shape == f.shape
