import pymorph
import numpy as np
def test_blob():
    F = np.array([
            [0,1,1,0,0],
            [0,2,2,0,0],
            [0,3,3,3,0],
            [4,4,4,4,4],
            ])

    M = pymorph.blob(F, 'area', 'data')
    MI = pymorph.blob(F, 'area', 'image')

    assert np.all( M == np.array([2,2,3,5]))
    for m,i in zip(MI.ravel(), F.ravel()):
        if i == 0: assert m == 0
        else: assert m == M[i-1]

    B = pymorph.blob(F, 'boundingbox', 'data')
    BI = pymorph.blob(F, 'boundingbox')
    assert BI.max() == 1

    C = pymorph.blob(F, 'centroid')
    assert np.all(C.sum(1) == 1)

