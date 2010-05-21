import pymorph
import numpy as np

def test_dilate():
    f = np.zeros((8,8), np.bool)
    Bs = [np.reshape(B, (3,3)) for B in (
                    [1,1,0, 1,1,0, 0,0,0],
                    [1,0,0, 1,1,0, 0,0,0],
                    [1,0,0, 0,1,0, 0,0,0],
                    [0,1,0, 0,1,0, 0,0,0],
                    )]
    for B in Bs:
        assert pymorph.dilate(f, B != 0).sum() == 0
        assert pymorph.dilate(f, B).sum() == 0
