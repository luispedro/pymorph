#!/usr/bin/env python
#-*- coding: ISO-8859-1 -*-

from matplotlib import pyplot, cm
import pymorph, numpy
import unittest
from os import path

_basedir = path.join(path.dirname(path.abspath(__file__)),'data')

class TestOpen(unittest.TestCase):
    def setUp(self):
        # img1.png has some letters around. b.png is one of them,
        # the 'b', extracted verbatim from img1.png.

        self.img = numpy.array(pyplot.imread(path.join(_basedir, 'img1.png')), dtype=numpy.bool_)
        self.ref = numpy.array(pyplot.imread(path.join(_basedir, 'b.png')), dtype=numpy.bool_)

    def test_brute_force(self):
        # Apply the definition and find the translations of b
        # that are included in img1.

        imgset = set()
        h, w = self.img.shape
        for y in xrange(h):
            for x in xrange(w):
                if self.img[y, x]:
                    imgset.add((y, x))

        refset = set()
        h, w = self.ref.shape
        for y in xrange(h):
            for x in xrange(w):
                if self.ref[y, x]:
                    refset.add((y, x))

        def isin(img, ref, dx, dy):
            r = set()
            for y, x in ref:
                r.add((y + dy, x + dx))
            return r.issubset(img)

        h, w = self.img.shape
        found = False

        for y in xrange(h):
            for x in xrange(w):
                if isin(imgset, refset, x, y):
                    self.assertEqual((x, y), (280, 85))
                    found = True
                    break
            if found:
                break
        else:
            self.fail('Not found')

    def test_pymorph(self):
        # Use pymorph.open()

        se = pymorph.img2se(self.ref)
        img = pymorph.open(self.img, se)

        expected = numpy.array(pyplot.imread(path.join(_basedir, 'expected.png')), dtype=numpy.bool_)

        h, w = img.shape
        for y in xrange(h):
            for x in xrange(w):
                if img[y, x] != expected[y, x]:
                    self.fail('Images differ at (%d, %d)' % (y, x))


if __name__ == '__main__':
    unittest.main()
