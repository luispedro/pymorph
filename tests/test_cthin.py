import pymorph
import numpy as np

def test_cthin():
    f = np.arange(16*16).reshape((16,16))%8
    g = (f > 2)
    f = (f > 3)
    t = pymorph.cthin(f,g)
    assert not np.any( ~g & t )

