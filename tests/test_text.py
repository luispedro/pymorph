import numpy as np
from pymorph import text, concat
def test_text_len():
    texts = ['o', 'on', 'one', 'one two', 'one two three']
    height = 15
    w_perch = 9
    for t in texts:
        assert text(t).shape == (height,w_perch*len(t))

def test_text_concat():
    o = text('o')
    n = text('n')
    e = text('e')
    assert o.shape == n.shape
    assert o.shape == e.shape
    assert np.all(concat('w', o, n, e,) == text('one'))
    assert np.all(concat('w', o, n, e,) == text('one'))
