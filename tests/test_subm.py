import pymorph
import numpy as np

def test_subm():
    assert not np.any( pymorph.subm(np.ones((3,3), np.bool), np.ones((3,3), np.bool)) )
    assert np.all( pymorph.subm(np.ones((32,33), np.bool), np.zeros((32,33), np.bool)) )
    assert np.all( pymorph.subm(np.ones((32,33), np.bool), np.zeros((32,33), np.uint8)) )
    assert np.all( pymorph.subm(np.ones((32,33), np.uint8)+15, np.zeros((32,33), np.uint8)) )
    assert not np.any( pymorph.subm(np.ones((3,3), np.uint8), np.ones((3,3), np.bool)) )
    assert not np.any( pymorph.subm(np.ones((3,3), np.uint8), 2+np.ones((3,3), np.uint8)) )
    assert np.all(pymorph.subm(np.ones((3,3), np.uint8), 2+np.ones((3,3), np.uint8))  == 0)
