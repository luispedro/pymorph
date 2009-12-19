import pymorph
import numpy as np

def test_serot():
    se = pymorph.secross()
    assert np.all(pymorph.serot(se, 90) == se)
    assert not np.all(pymorph.serot(se, 45) == se)

