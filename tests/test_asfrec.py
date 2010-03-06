import pymorph
import numpy as np
def test_asfrec():
    f = np.array([
        [0,1,0,0],
        [1,0,1,1],
        [1,1,1,1],
        [0,0,0,1],
        ],bool)
    assert pymorph.asfrec(f).shape == f.shape
    assert pymorph.asfrec(f).all()
    assert not pymorph.asfrec(f,'CO').all()
