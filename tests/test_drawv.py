import pymorph
import numpy as np

def test_drawv():
    Point = pymorph.drawv(np.zeros((8,8), np.uint8), [(1,1),(6,6)], 123, 'point')
    assert np.all( (Point == 0) | (Point == 123) )
    assert Point[1,1] == 123
    assert Point[6,6] == 123
    assert (Point == 123).sum() == 2

    Line = pymorph.drawv(np.zeros((8,8), np.uint8), [(1,1),(6,6)], 123, 'line')
    assert np.all( (Line == 0) | (Line == 123) )
    assert np.all( (Line == 0) | (Line == 123) )
    assert (Line.diagonal() == 123).sum() == 6
    assert (Line == 123).sum() == 6

    Rect = pymorph.drawv(np.zeros((8,8), np.uint8), [(1,1,6,6)], 123, 'rect')
    assert np.all( (Rect == 123) | (Rect == 0) )
    assert (Rect == 123).sum() == (6*4-4)

    Frect = pymorph.drawv(np.zeros((8,8), np.uint8), [(1,1,6,6)], 123, 'frect')
    assert np.all( (Rect == 123) | (Rect == 0) )
    assert (Frect == 123).sum() == 6*6

