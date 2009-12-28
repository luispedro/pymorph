from pymorph import hmax
import numpy as np
def test_hmax():
    a = np.array([1,1,1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,1,1,1,], np.uint8)
    assert np.all(hmax(a, a.max()) == 0)
    for i in xrange(1,5):
        assert hmax(a,i).max() == a.max()-i


