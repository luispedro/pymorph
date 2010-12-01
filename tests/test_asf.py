import pymorph
import numpy as np

def test_OCO():
    # In 0.95, 3 letter modes crashed
    # reported by Alexandre Harano
    f = np.arange(16*16).reshape((16,16))%8
    for mode in 'OC', 'CO', 'COC', 'OCO':
        g = pymorph.asf(f, mode)
        assert g.shape == f.shape

