import pymorph
import pylab
pieces = pylab.imread('pymorph/data/pieces_bw.tif').max(2)
h,w = pieces.shape

def test_isolines():
    D = pymorph.dist(pieces == 0)
    assert D.shape == (h,w)
    assert pymorph.isolines(D).shape == (h,w,3)

def test_randomcolor():
    assert (h,w,3) == pymorph.randomcolor(pymorph.label(pieces > 0)).shape

def test_label():
    assert (h,w) == pymorph.label(pieces > 0).shape

def test_overlay():
    img = pylab.imread('pymorph/data/fabric.tif')
    assert pymorph.overlay(img, img == 255).shape == (img.shape + (3,))
