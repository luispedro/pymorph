import pymorph
import numpy as np
def _brute_test(se):
    on,_ = pymorph.mat2set(se)
    off,_ = pymorph.mat2set(pymorph.sereflect(se))
    assert set(map(tuple,map(np.negative, on))) == set(map(tuple,off))
def test_sereflect():
    yield _brute_test, pymorph.secross()

    se = pymorph.secross()
    se[1,1] = 1 # This makes it non-symmetric
    yield _brute_test, se

    se[2,1] = 1 # This makes it non-symmetric
    yield _brute_test, se

    se = np.zeros((9,9), bool)
    se[0,4] = 1
    yield _brute_test, se

    se = np.zeros((9,9), bool)
    se[1,1] = se[1,2] = se[0,4] = 1
    yield _brute_test, se
