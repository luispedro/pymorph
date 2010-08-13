import numpy as np
import pymorph
def test_toggle():
    f = np.arange(20)
    f1 = np.arange(20)//5
    f2 = 4-np.arange(20)//5
    assert pymorph.toggle(f, f1, f2, True).shape == f.shape
    assert pymorph.toggle(f, f1, f2, False).shape == f.shape

