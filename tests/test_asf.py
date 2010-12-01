import pymorph
import numpy as np

def test_OCO():
    # In 0.95, this crashed.
    # reported by Alexandre Harano
    f = np.arange(16*16).reshape((16,16))%8
    pymorph.asf(f, 'OCO')

