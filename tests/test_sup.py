import numpy as np
from pymorph import mmorph
f = np.array([
  [0,0,1,0,0,1,1],
  [0,1,0,0,1,0,0],
  [0,0,0,1,1,0,0]],np.bool)
i = mmorph.endpoints()

def test_supcanon():
    g = mmorph.supgen(f,i)
    assert g.sum() == 1
def test_supcanon():
    g = mmorph.supcanon(f,i)
    assert np.all((f|g) == f)

    g = mmorph.supcanon(f,i, 90)
    assert np.all((f|g) == f)

    g = mmorph.supcanon(f,i, 180)
    assert np.all((f|g) == f)

