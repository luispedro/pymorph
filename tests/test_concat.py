import pymorph
import numpy as np
def test_concat():
    np.random.seed(123)
    A0 = np.random.rand(2,3)
    A1 = np.random.rand(2,3)
    A2 = np.random.rand(2,3)
    A3 = np.random.rand(2,3)
    A0123 = pymorph.concat('d',A0,A1,A2,A3)
    assert A0123.shape == (2,3,4)
    assert np.all(A0123[:,:,0] == A0)
    assert np.all(A0123[:,:,2] == A2)

    A0123 = pymorph.concat('h',A0,A1,A2,A3)
    assert A0123.shape == (2*4,3)
    assert np.all(A0123[:2] == A0)
    assert np.all(A0123[-2:] == A3)

    A0123 = pymorph.concat('w',A0,A1,A2,A3)
    assert A0123.shape == (2,3*4)
    assert np.all(A0123[:,-3:] == A3)
    assert np.all(A0123[:,:3] == A0)

def test_concat_different_sizes():
    np.random.seed(123)
    A0 = np.random.rand(2,3)
    A1 = np.random.rand(2,4)
    A2 = np.random.rand(2,5)
    A3 = np.random.rand(2,6)

    A0123 = pymorph.concat('d',A0,A1,A2,A3)

    assert A0123.shape == (2,6,4)
