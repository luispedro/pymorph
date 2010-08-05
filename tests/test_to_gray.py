import pymorph
import pylab
import numpy as np
def test_to_gray():
    pieces = pylab.imread('pymorph/data/pieces_bw.tif')
    assert np.all(pymorph.to_gray(pieces) == pieces[:,:,:3].max(2))

def test_2d():
    img = pylab.imread('pymorph/data/fabric.tif')
    assert img.shape == pymorph.to_gray(img).shape
    assert np.all(img == pymorph.to_gray(img))

