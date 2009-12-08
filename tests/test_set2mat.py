import numpy as np
from pymorph import to_int32, set2mat

def test_set2mat():
    coord = to_int32([
                  [ 0,0],
                  [-1,0],
                  [ 1,1]])
    A = set2mat((coord,))
    for x,y in coord:
        assert A[x+1, y+1] == 1
    for x,y in coord:
        A[x+1, y+1] = 0
    assert not A.any()

