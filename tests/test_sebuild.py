import pymorph
import numpy as np

def test_secross():
    assert np.all(pymorph.secross() == np.array([[0,1,0],[1,1,1],[0,1,0]]))
    assert np.all(pymorph.secross(0) == np.array([[1]]))
    assert np.all(pymorph.secross(2).shape == (5,5))
    assert np.all(pymorph.secross(9).shape == (2*9+1,2*9+1))

def test_sebox():
    assert np.all(pymorph.sebox(0) == np.array([1]))
    assert np.all(pymorph.sebox(1) == np.ones((3,3)))
    assert np.all(pymorph.sebox(9) == np.ones((2*9+1,2*9+1)))
