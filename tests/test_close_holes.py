import numpy as np
import pymorph
np.random.seed(1222)

def test_close_holes_random():
    H = (np.random.rand(100,100) > .2)
    Hclosed = pymorph.close_holes(H)
    assert not (H & ~Hclosed).any()

def test_close_holes_simple():
    H = pymorph.binary([
                [0,0,0,0,0,0],
                [0,1,1,1,1,0],
                [0,1,0,0,1,0],
                [0,1,1,1,1,0],
                [0,0,0,0,0,0],
            ])

    Hclosed = pymorph.close_holes(H)
    assert np.all(Hclosed  == pymorph.binary([
                    [0,0,0,0,0,0],
                    [0,1,1,1,1,0],
                    [0,1,1,1,1,0],
                    [0,1,1,1,1,0],
                    [0,0,0,0,0,0],
                    ]))

