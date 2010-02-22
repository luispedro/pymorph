import numpy as np
import pymorph

def test_labelflat():
    assert np.all(pymorph.labelflat(pymorph.secross()) == pymorph.secross())

