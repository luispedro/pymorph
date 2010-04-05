import pymorph
import numpy as np

def test_skiz():
    f = np.array([
            [0,0,1,1,1],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [1,0,0,0,0]])
    labeled,lines = pymorph.skiz(f!= 0, return_lines=1)
    assert labeled.max() == 2
    assert labeled.min() == 1
    Y,X = np.where(lines)
    assert Y.size
    for y,x in zip(Y,X):
        pos = []
        for dy in (-1, 0, +1):
            for dx in (-1, 0, +1):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < f.shape[0] and 0 <= nx < f.shape[1]:
                    pos.append(labeled[ny,nx])
        assert len(set(pos)) > 1
