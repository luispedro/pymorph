import numpy as np
import pymorph

def test_frame():
    F10 = pymorph.frame(np.zeros((10,10), bool))
    assert F10.any()
    assert not F10.all()
    assert F10.sum() == (10*4 - 4) # four sides of length 10, but don't double count 4 corners.

    F5 = pymorph.frame(np.zeros((5,5), bool))
    assert np.all( F5 == np.array([
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,1,1,1,1],
        ]) )
