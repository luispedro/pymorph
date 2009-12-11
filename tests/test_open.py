from matplotlib import pyplot
import pymorph
import numpy as np
from os import path

_basedir = path.join(path.dirname(path.abspath(__file__)),'data')

def test_open():
    img = np.array(pyplot.imread(path.join(_basedir, 'img1.png')), dtype=bool)
    ref = np.array(pyplot.imread(path.join(_basedir, 'b.png')), dtype=bool)
    expected = np.array(pyplot.imread(path.join(_basedir, 'expected.png')), dtype=bool)

    se = pymorph.img2se(ref)
    res = pymorph.open(img, se)

    assert np.all(img[85:99,280:289] == ref)
    assert np.all(res == expected)

