import pymorph
def test_isolines():
    import readmagick
    pieces = readmagick.readimg('pymorph/data/pieces_bw.tif').max(2)
    D = pymorph.dist(pieces == 0)
    h,w = D.shape
    assert pymorph.isolines(D).shape == (h,w,3)

