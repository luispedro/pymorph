from pymorph import supgen
import numpy as np
def test_supgen():
    f = np.zeros((4,4), np.bool)
    f[3,2] = 1
    f[3,3] = 1
    f[2,2] = 1
    interval = (np.reshape([1,0,0,1],(2,2)), np.reshape([0,1,0,0],(2,2))) 
    match = supgen(f, interval)
    assert match.sum() == 1
