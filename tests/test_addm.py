from pymorph import to_uint8, addm

def test_addm():
    f = to_uint8([255,   255,    0,   10,    0,   255,   250])
    g = to_uint8([ 0,    40,   80,   140,  250,    10,    30])
    y = addm(f,g)
    for fi,gi,yi in zip(f,g,y):
        assert yi >= fi
        assert yi >= gi

