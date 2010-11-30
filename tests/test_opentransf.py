import pymorph
import numpy as np
def test_opentransf():
    f = np.array([
            [0,0,0,0,0,0,0,0],
            [0,0,1,1,1,1,0,0],
            [0,1,0,1,1,1,0,0],
            [0,0,1,1,1,1,0,0],
            [1,1,0,0,0,0,0,0]], bool)
    ot = pymorph.opentransf( f, 'city-block')
    for y in xrange(ot.shape[0]):
        for x in xrange(ot.shape[1]):
            r = ot[y,x]
            t = f.copy()
            for k in xrange(1, r+1):
                assert t[y,x]
                t = pymorph.open(f, pymorph.sedisk(k, 2, 'city-block'))
            assert not t[y,x]

def test_all_types():
    f = np.arange(9).reshape((3,3)) % 3 > 0
    # linear-h crashed in 0.95
    # and was underlying cause of crash in patsec(f, 'linear-h')

    def test_type(type, Buser):
        g = pymorph.opentransf(f, type, Buser=Buser)

    yield test_type, 'linear-h', None
    yield test_type, 'octagon', None
    yield test_type, 'chessboard', None
    yield test_type, 'city-block', None
    yield test_type, 'linear-v', None
    yield test_type, 'linear-45r', None
    yield test_type, 'linear-45l', None
    Buser = np.ones((3,3),bool)
    Buser[2,2] = 0
    yield test_type, 'user', Buser
