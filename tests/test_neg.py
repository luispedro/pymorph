import pymorph
def test_neg():
    pymorph.neg(pymorph.secross()) == ~pymorph.secross()

