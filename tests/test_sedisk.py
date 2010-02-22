import numpy as np
import pymorph

def test_sedisk():
    se = pymorph.sedisk(4)
    w,h = se.shape
    assert w == h
    X,Y = np.where(~se)
    X -= w//2
    Y -= h//2
    assert np.all( X**2+Y**2 > (w//2)**2 )

