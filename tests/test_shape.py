import pymorph
import readmagick
pieces = readmagick.readimg('pymorph/data/pieces_bw.tif').max(2)
h,w = pieces.shape

def test_isolines():
    import readmagick
    D = pymorph.dist(pieces == 0)
    assert D.shape == (h,w)
    assert pymorph.isolines(D).shape == (h,w,3)

def test_randomcolor():
    assert (h,w,3) == pymorph.randomcolor(pymorph.label(pieces > 0)).shape

def test_label():
    assert (h,w) == pymorph.label(pieces > 0).shape

