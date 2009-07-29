import pymorph
import readmagick
import numpy as np
def test_to_gray():
    pieces = readmagick.readimg('pymorph/data/pieces_bw.tif')
    assert np.all(pymorph.to_gray(pieces) == pieces.max(2))
