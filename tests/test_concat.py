from pymorph import concat
import numpy as np

def test_concat():
    a = np.arange(24).reshape((6,4))
    b = np.arange(24).reshape((6,4))**2
    c = np.arange(24).reshape((6,4))*3
    assert np.all(concat('w', a, concat('w', b, c,)) == concat('w', a, b, c))
    assert np.all(concat('w', concat('w', a, b,), c) == concat('w', a, b, c))
    assert concat('w',a,b).shape == (6,8)
    assert concat('w',a,b,c).shape == (6,12)

