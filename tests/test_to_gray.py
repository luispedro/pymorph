import pymorph
import readmagick
import numpy as np
def test_to_gray():
    pieces = readmagick.readimg('pymorph/data/pieces_bw.tif')
    assert np.all(pymorph.to_gray(pieces) == pieces.max(2))

def test_2d():
    img = readmagick.readimg('pymorph/data/fabric.tif')
    assert img.shape == pymorph.to_gray(img).shape
    assert np.all(img == pymorph.to_gray(img))

