import pymorph 
import numpy as np
def test_grain():
    L = np.array([
        [0,0,1,1,1,0],
        [0,1,0,0,0,0],
        [0,0,2,2,2,0],
        [0,2,2,2,2,2],
        ])
    D = np.array([
        [0, 0,12,12,12,0],
        [0,12, 0, 0, 0,0],
        [0,0,21,22,23,0],
        [0,24,25,26,27,28],
        ])

    assert np.all( pymorph.grain(D, L, 'max', 'data') == (12, 28))
    assert np.all( pymorph.grain(D, L, 'min', 'data') == (12, 21))
    assert np.all( pymorph.grain(D, L, 'mean', 'data') == (12, 24.5))

