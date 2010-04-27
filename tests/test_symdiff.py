import numpy as np
from pymorph import mmorph
def test_symdiff():
    f = np.array([
        [0,1,1],
        [1,1,0]],np.bool)
    g = np.array([
        [0,1,0],
        [1,1,0]], np.bool)
    assert mmorph.symdif(f,~f).sum() == f.size
    h = mmorph.symdif(f,g)
    assert np.all(h == ( (f&~g) | (g&~f) ))

