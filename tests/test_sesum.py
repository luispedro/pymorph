import pymorph
import numpy as np

def test_sesum():
    assert np.all(pymorph.sesum(pymorph.secross(), 1) == pymorph.secross())
    assert len(pymorph.sesum(pymorph.secross(), 0).shape) == 2

