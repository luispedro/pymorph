from pymorph.mmorph import intershow
import numpy as np
def test_intershow():
    A = np.array([
            [0,1],
            [1,0]])
    B = np.array([
            [1,0],
            [0,0]])
    res = '0 1 \n1 . \n'
    assert intershow((A, B)) == res
    assert intershow((A != 0, B != 0)) == res

