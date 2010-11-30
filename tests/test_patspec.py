import pymorph
import numpy as np
def test_patspec():
    f = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,0,0],
        [0,1,0,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [1,1,0,0,0,0,0,0]], bool)
    assert pymorph.patspec(f).sum() == (f > 0).sum()

def test_linear_h():
    f = np.arange(9).reshape((3,3)) % 3 > 0
    # This crashed in 0.95
    # reported by Alexandre Harano
    g = pymorph.patspec(f, 'linear-h')

