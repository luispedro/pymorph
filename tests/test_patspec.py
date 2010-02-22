import pymorph
import numpy as np
def test_patspec():
    f = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,0,0],
        [0,1,0,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [1,1,0,0,0,0,0,0]], bool)
    assert pymorph.patspec(f).sum() == (f > 0).sum()

