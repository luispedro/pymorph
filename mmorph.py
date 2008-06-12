"""
    Module morph -- SDC Morphology Toolbox
    -------------------------------------------------------------------
    The pymorph Morphology Toolbox for Python is a powerful collection of latest
    state-of-the-art gray-scale morphological tools that can be applied to image
    segmentation, non-linear filtering, pattern recognition and image analysis.
    -------------------------------------------------------------------
    int32()          -- Convert an image to an int32 image.
    mmadd4dil()      -- Addition for dilation
    mmaddm()         -- Addition of two images, with saturation.
    mmareaclose()    -- Area closing
    mmareaopen()     -- Area opening
    mmasf()          -- Alternating Sequential Filtering
    mmasfrec()       -- Reconstructive Alternating Sequential Filtering
    mmbench()        -- benchmarking main functions of the toolbox.
    mmbinary()       -- Convert a gray-scale image into a binary image
    mmblob()         -- Blob measurements from a labeled image.
    mmbshow()        -- Generate a graphical representation of overlaid binary
                        images.
    mmcbisector()    -- N-Conditional bisector.
    mmcdil()         -- Dilate an image conditionally.
    mmcenter()       -- Center filter.
    mmcero()         -- Erode an image conditionally.
    mmclohole()      -- Close holes of binary and gray-scale images.
    mmclose()        -- Morphological closing.
    mmcloserec()     -- Closing by reconstruction.
    mmcloserecth()   -- Close-by-Reconstruction Top-Hat.
    mmcloseth()      -- Closing Top Hat.
    mmcmp()          -- Compare two images pixelwisely.
    mmconcat()       -- Concatenate two or more images along width, height or
                        depth.
    mmcthick()       -- Image transformation by conditional thickening.
    mmcthin()        -- Image transformation by conditional thinning.
    mmcwatershed()   -- Detection of watershed from markers.
    mmdatatype()     -- Return the image datatype string
    mmdil()          -- Dilate an image by a structuring element.
    mmdist()         -- Distance transform.
    mmdrawv()        -- Superpose points, rectangles and lines on an image.
    mmdtshow()       -- Display a distance transform image with an iso-line
                        color table.
    mmedgeoff()      -- Eliminate the objects that hit the image frame.
    mmendpoints()    -- Interval to detect end-points.
    mmero()          -- Erode an image by a structuring element.
    mmflood()        -- Flooding filter- h,v,a-basin and dynamics (depth, area,
                        volume)
    mmframe()        -- Create a frame image.
    mmfreedom()      -- Control automatic data type conversion.
    mmgdist()        -- Geodesic Distance Transform.
    mmgdtshow()      -- Apply an iso-line color table to a gray-scale image.
    mmglblshow()     -- Apply a random color table to a gray-scale image.
    mmgradm()        -- Morphological gradient.
    mmgrain()        -- Gray-scale statistics for each labeled region.
    mmgray()         -- Convert a binary image into a gray-scale image.
    mmgshow()        -- Apply binary overlays as color layers on a binary or
                        gray-scale image
    mmhistogram()    -- Find the histogram of the image f.
    mmhmax()         -- Remove peaks with contrast less than h.
    mmhmin()         -- Remove basins with contrast less than h.
    mmhomothick()    -- Interval for homotopic thickening.
    mmhomothin()     -- Interval for homotopic thinning.
    mmimg2se()       -- Create a structuring element from a pair of images.
    mminfcanon()     -- Intersection of inf-generating operators.
    mminfgen()       -- Inf-generating.
    mminfrec()       -- Inf-reconstruction.
    mminpos()        -- Minima imposition.
    mminstall()      -- Verify if the Morphology Toolbox is registered.
    mminterot()      -- Rotate an interval
    mmintersec()     -- Intersection of images.
    mmintershow()    -- Visualize an interval.
    mmis()           -- Verify if a relationship among images is true or false.
    mmisbinary()     -- Check for binary image
    mmisequal()      -- Verify if two images are equal
    mmislesseq()     -- Verify if one image is less or equal another (is
                        beneath)
    mmlabel()        -- Label a binary image.
    mmlabelflat()    -- Label the flat zones of gray-scale images.
    mmlastero()      -- Last erosion.
    mmlblshow()      -- Display a labeled image assigning a random color for
                        each label.
    mmlimits()       -- Get the possible minimum and maximum of an image.
    mmmat2set()      -- Converts image representation from matrix to set
    mmmaxleveltype() -- Returns the maximum value associated to an image
                        datatype
    mmneg()          -- Negate an image.
    mmopen()         -- Morphological opening.
    mmopenrec()      -- Opening by reconstruction.
    mmopenrecth()    -- Open-by-Reconstruction Top-Hat.
    mmopenth()       -- Opening Top Hat.
    mmopentransf()   -- Open transform.
    mmpad4n()        -- mmpad4n
    mmpatspec()      -- Pattern spectrum (also known as granulometric size
                        density).
    mmplot()         -- Plot a function.
    mmreadgray()     -- Read an image from a commercial file format and stores
                        it as a gray-scale image.
    mmregister()     -- Register the SDC Morphology Toolbox.
    mmregmax()       -- Regional Maximum.
    mmregmin()       -- Regional Minimum (with generalized dynamics).
    mmse2hmt()       -- Create a Hit-or-Miss Template (or interval) from a pair
                        of structuring elements.
    mmse2interval()  -- Create an interval from a pair of structuring elements.
    mmsebox()        -- Create a box structuring element.
    mmsecross()      -- Diamond structuring element and elementary 3x3 cross.
    mmsedil()        -- Dilate one structuring element by another
    mmsedisk()       -- Create a disk or a semi-sphere structuring element.
    mmseline()       -- Create a line structuring element.
    mmsereflect()    -- Reflect a structuring element
    mmserot()        -- Rotate a structuring element.
    mmseshow()       -- Display a structuring element as an image.
    mmsesum()        -- N-1 iterative Minkowski additions
    mmset2mat()      -- Converts image representation from set to matrix
    mmsetrans()      -- Translate a structuring element
    mmseunion()      -- Union of structuring elements
    mmshow()         -- Display binary or gray-scale images and optionally
                        overlay it with binary images.
    mmskelm()        -- Morphological skeleton (Medial Axis Transform).
    mmskelmrec()     -- Morphological skeleton reconstruction (Inverse Medial
                        Axis Transform).
    mmskiz()         -- Skeleton of Influence Zone - also know as Generalized
                        Voronoi Diagram
    mmstats()        -- Find global image statistics.
    mmsubm()         -- Subtraction of two images, with saturation.
    mmsupcanon()     -- Union of sup-generating or hit-miss operators.
    mmsupgen()       -- Sup-generating (hit-miss).
    mmsuprec()       -- Sup-reconstruction.
    mmswatershed()   -- Detection of similarity-based watershed from markers.
    mmsymdif()       -- Symmetric difference between two images
    mmtext()         -- Create a binary image of a text.
    mmthick()        -- Image transformation by thickening.
    mmthin()         -- Image transformation by thinning.
    mmthreshad()     -- Threshold (adaptive)
    mmtoggle()       -- Image contrast enhancement or classification by the
                        toggle operator.
    mmunion()        -- Union of images.
    mmvdome()        -- Obsolete, use mmvmax.
    mmversion()      -- SDC Morphology Toolbox version.
    mmvmax()         -- Remove domes with volume less than v.
    mmwatershed()    -- Watershed detection.
    uint16()         -- Convert an image to a uint16 image.
    to_uint8()          -- Convert an image to an uint8 image.

    ---

"""
#
__version__ = '0.8 pybase'

__version_string__ = 'SDC Morphology Toolbox V0.8 01Aug03 (script version)'

__build_date__ = '04aug2003 12:07'
#

import sys, os
mydir = os.path.dirname(__file__)
try:
    sys.imagepath += [os.path.join(mydir, 'data')]
except:
    sys.imagepath = [os.path.join(mydir, 'data')]

#

#
# =====================================================================
#
#   mmconcat
#
# =====================================================================
def mmconcat(DIM, X1, X2, X3=None, X4=None):
    """
        - Purpose
            Concatenate two or more images along width, height or depth.
        - Synopsis
            Y = mmconcat(DIM, X1, X2, X3=None, X4=None)
        - Input
            DIM: String Dimension to concatenate. 'WIDTH' or 'W', 'HEIGHT'
                 or 'H', or ' DEPTH' or 'D'.
            X1:  Gray-scale (uint8 or uint16) or binary image.
            X2:  Gray-scale (uint8 or uint16) or binary image.
            X3:  Gray-scale (uint8 or uint16) or binary image. Default:
                 None.
            X4:  Gray-scale (uint8 or uint16) or binary image. Default:
                 None.
        - Output
            Y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            Concatenate two or more images in any of the dimensions: width,
            height or depth. If the images do not match the dimension, a
            larger image is create with zero pixels to accommodate them. The
            images must have the same datatype.
        - Examples
            #
            f1=mmreadgray('cameraman.tif')
            f2=mmreadgray('blob.tif')
            g=mmconcat('W',f1,mmgray(mmneg(f2)))
            mmshow(g);
    """
    from numpy import newaxis, sum, zeros

    aux = 'newaxis,'
    d = len(X1.shape)
    if d < 3: X1 = eval('X1[' + (3-d)*aux + ':]')
    d1,h1,w1 = X1.shape
    d = len(X2.shape)
    if d < 3: X2 = eval('X2[' + (3-d)*aux + ':]')
    d2,h2,w2 = X2.shape
    h3 = w3 = d3 = h4 = w4 = d4 = 0
    if X3:
       d = len(X3.shape)
       if d < 3: X3 = eval('X3[' + (3-d)*aux + ':]')
       d3,h3,w3 = X3.shape
    if X4:
       d = len(X4.shape)
       if d < 3: X4 = eval('X4[' + (3-d)*aux + ':]')
       d4,h4,w4 = X4.shape
    h = [h1, h2, h3, h4]
    w = [w1, w2, w3, w4]
    d = [d1, d2, d3, d4]
    if DIM in ['WIDTH', 'W', 'w', 'width']:
       hy, wy, dy = max(h), sum(w), max(d)
       Y = zeros((dy,hy,wy))
       Y[0:d1, 0:h1, 0 :w1   ] = X1
       Y[0:d2, 0:h2, w1:w1+w2] = X2
       if X3:
          Y[0:d3, 0:h3, w1+w2:w1+w2+w3] = X3
          if X4:
              Y[0:d4, 0:h4, w1+w2+w3::] = X4
    elif DIM in ['HEIGHT', 'H', 'h', 'height']:
       hy, wy, dy = sum(h), max(w), max(d)
       Y = zeros((dy,hy,wy))
       Y[0:d1, 0 :h1   , 0:w1] = X1
       Y[0:d2, h1:h1+h2, 0:w2] = X2
       if X3:
           Y[0:d3, h1+h2:h1+h2+h3, 0:w3] = X3
           if X4:
               Y[0:d4, h1+h2+h3::, 0:w4] = X4
    elif DIM in ['DEPTH', 'D', 'd', 'depth']:
       hy, wy, dy = max(h), max(w), sum(d)
       Y = zeros((dy,hy,wy))
       Y[0:d1    , 0:h1, 0:w1   ] = X1
       Y[d1:d1+d2, 0:h2, 0:w2] = X2
       if X3:
           Y[d1+d2:d1+d2+d3, 0:h3, 0:w3] = X3
           if X4:
               Y[d1+d2+d3::, 0:h4, 0:w4] = X4
    if Y.shape[0] == 1: # adjustment
       Y = Y[0,:,:]
    return Y
#
# =====================================================================
#
#   mmlimits
#
# =====================================================================
def mmlimits(f):
    """
        - Purpose
            Get the possible minimum and maximum of an image.
        - Synopsis
            y = mmlimits(f)
        - Input
            f: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image.
        - Output
            y: Vector, the first element is the infimum, the second, the
               supremum.
        - Description
            The possible minimum and the possible maximum of an image depend
            on its data type. These values are important to compute many
            morphological operators (for instance, negate of an image). The
            output is a vector, where the first element is the possible
            minimum and the second, the possible maximum.
        - Examples
            #
            print mmlimits(mmbinary([0, 1, 0]))
            print mmlimits(to_uint8([0, 1, 2]))
    """
    from numpy import array

    import numpy as N
    code = f.dtype
    if   code == N.bool: y=array([0,1])
    elif code == N.uint8: y=array([0,255])
    elif code == N.uint16: y=array([0,65535])
    elif code == N.int32: y=array([-2147483647,2147483647])
    else:
        assert 0,'Does not accept this typecode:'+code
    return y
#
# =====================================================================
#
#   mmcenter
#
# =====================================================================
def mmcenter(f, b=None):
    """
        - Purpose
            Center filter.
        - Synopsis
            y = mmcenter(f, b=None)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            b: Structuring Element Default: None (3x3 elementary cross).
        - Output
            y: Image
        - Description
            mmcenter creates the image y by computing recursively the
            morphological center, relative to the structuring element b , of
            the image f .
        - Examples
            #
            f=mmreadgray('gear.tif')
            g=mmcenter(f,mmsedisk(2))
            mmshow(f)
            mmshow(g)
    """

    if b is None: b = mmsecross()
    y = f
    diff = 0
    while not diff:
        aux = y
        beta1 = mmasf(y,'COC',b,1)
        beta2 = mmasf(y,'OCO',b,1)
        y = mmunion(mmintersec(y,beta1),beta2)
        diff = mmisequal(aux,y)
    return y
#
# =====================================================================
#
#   mmclohole
#
# =====================================================================
def mmclohole(f, Bc=None):
    """
        - Purpose
            Close holes of binary and gray-scale images.
        - Synopsis
            y = mmclohole(f, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) or binary image.
            Bc: Structuring Element Default: None (3x3 elementary cross). (
                connectivity).
        - Output
            y: (same datatype of f ).
        - Description
            mmclohole creates the image y by closing the holes of the image
            f , according with the connectivity defined by the structuring
            element Bc .The images can be either binary or gray-scale.
        - Examples
            #
            #   example 1
            #
            a = mmreadgray('pcb1bin.tif')
            b = mmclohole(a)
            mmshow(a)
            mmshow(b)
            #
            #   example 2
            #
            a = mmreadgray('boxdrill-B.tif')
            b = mmclohole(a)
            mmshow(a)
            mmshow(b)
    """

    if Bc is None: Bc = mmsecross()
    delta_f = mmframe(f)
    y = mmneg( mminfrec( delta_f, mmneg(f), Bc))
    return y
#
# =====================================================================
#
#   mmdist
#
# =====================================================================
def mmdist(f, Bc=None, METRIC=None):
    """
        - Purpose
            Distance transform.
        - Synopsis
            y = mmdist(f, Bc=None, METRIC=None)
        - Input
            f:      Binary image.
            Bc:     Structuring Element Default: None (3x3 elementary
                    cross). (connectivity)
            METRIC: String Default: None. 'EUCLIDEAN', or 'EUC2' for squared
                    Euclidean.
        - Output
            y: distance image in uint16, or in int32 datatype with EUC2
               option.
        - Description
            mmdist creates the distance image y of the binary image f . The
            value of y at the pixel x is the distance of x to the complement
            of f , that is, the distance of x to nearest point in the
            complement of f . The distances available are based on the
            Euclidean metrics and on metrics generated by a a regular graph,
            that is characterized by a connectivity rule defined by the
            structuring element Bc . The implementation of the Euclidean
            algorithm is based on LotuZamp:01 .
        - Examples
            #
            #   example 1
            #
            a = mmframe(mmbinary(ones((5,9))),2,4)
            f4=mmdist(a)
            f8=mmdist(a,mmsebox())
            fe=mmdist(a,mmsebox(),'EUCLIDEAN')
            #
            #   example 2
            #
            f = mmreadgray('gear.tif')
            f = mmneg(mmgradm(f))
            d4=mmdist(f)
            d8=mmdist(f,mmsebox())
            de=mmdist(f,mmsebox(),'EUCLIDEAN')
            mmshow(f)
            mmshow(d4%8)
            mmshow(d8%8)
            mmshow(de%8)
    """
    from string import upper
    from numpy import zeros, sqrt
    if Bc is None: Bc = mmsecross()
    if METRIC is not None:
       METRIC = upper(METRIC)
    f = mmgray(f,'uint16')
    y = mmintersec(f,0)
    if (METRIC == 'EUCLIDEAN') or (METRIC == 'EUC2'):
        b = int32(zeros((3,3)))
        i=1
        while not mmisequal(f,y):
            a4,a2 = -4*i+2, -2*i+1
            b = int32([[a4,a2,a4],
                       [a2, 0,a2],
                       [a4,a2,a4]])
            y=f
            i=i+1
            f = mmero(f,b)
        if METRIC == 'EUCLIDEAN':
            f = uint16(sqrt(f)+0.5)
    else:
        if mmisequal(Bc, mmsecross()):
            b = int32([[-2147483647,  -1, -2147483647],
                       [         -1,   0,          -1],
                       [-2147483647,  -1, -2147483647]])
        elif mmisequal(Bc, mmsebox()):
            b = int32([[-1,-1,-1],
                       [-1, 0,-1],
                       [-1,-1,-1]])
        else: b = Bc
        while not mmisequal(f,y):
            y=f
            f = mmero(f,b)
    return y
#
# =====================================================================
#
#   mmedgeoff
#
# =====================================================================
def mmedgeoff(f, Bc=None):
    """
        - Purpose
            Eliminate the objects that hit the image frame.
        - Synopsis
            y = mmedgeoff(f, Bc=None)
        - Input
            f:  Binary image.
            Bc: Structuring Element Default: None (3x3 elementary cross). (
                connectivity)
        - Output
            y: Binary image.
        - Description
            mmedgeoff creates the binary image y by eliminating the objects
            (connected components) of the binary image f that hit the image
            frame, according to the connectivity defined by the structuring
            element Bc .
        - Examples
            #
            a=mmreadgray('form-1.tif')
            b=mmedgeoff(a)
            mmshow(a)
            mmshow(b)
    """

    if Bc is None: Bc = mmsecross()
    edge = mmframe(f)
    y = mmsubm( f, mminfrec(edge, f, Bc))
    return y
#
# =====================================================================
#
#   mmframe
#
# =====================================================================
def mmframe(f, WT=1, HT=1, DT=0, k1=None, k2=None):
    """
        - Purpose
            Create a frame image.
        - Synopsis
            y = mmframe(f, WT=1, HT=1, DT=0, k1=None, k2=None)
        - Input
            f:  Unsigned gray-scale (uint8 or uint16), signed (int32) or
                binary image.
            WT: Double Default: 1. Positive integer ( width thickness).
            HT: Double Default: 1. Positive integer ( height thickness).
            DT: Double Default: 0. Positive integer ( depth thickness).
            k1: Non-negative integer. Default: None (Maximum pixel value
                allowed in f). Frame gray-level.
            k2: Non-negative integer. Default: None (Minimum pixel value
                allowed in f). Background gray level.
        - Output
            y: image of same type as f .
        - Description
            mmframe creates an image y , with the same dimensions (W,H,D)
            and same pixel type of the image f , such that the value of the
            pixels in the image frame is k1 and the value of the other
            pixels is k2 . The thickness of the image frame is DT.

    """

    if k1 is None: k1 = mmlimits(f)[1]
    if k2 is None: k2 = mmlimits(f)[0]
    assert len(f.shape)<3,'Supports 2D only'
    y = mmunion(mmintersec(f,mmlimits(f)[0]),k2)
    y[:,0:WT] = k1
    y[:,-WT:] = k1
    y[0:HT,:] = k1
    y[-HT:,:] = k1
    return y
#
# =====================================================================
#
#   mmglblshow
#
# =====================================================================
def mmglblshow(X, border=0.0):
    """
        - Purpose
            Apply a random color table to a gray-scale image.
        - Synopsis
            Y = mmglblshow(X, border=0.0)
        - Input
            X:      Gray-scale (uint8 or uint16) image. Labeled image.
            border: Boolean Default: 0.0. Labeled image.
        - Output
            Y: Gray-scale (uint8 or uint16) or binary image.

    """
    from numpy import take, resize, shape
    from MLab import rand

    mmin = mmstats(X,'min')
    mmax = mmstats(X,'max')
    ncolors = mmax - mmin + 1
    R = int32(rand(ncolors)*255)
    G = int32(rand(ncolors)*255)
    B = int32(rand(ncolors)*255)
    if mmin == 0:
       R[0],G[0],B[0] = 0,0,0
    r=resize(take(R, X.flat - mmin),X.shape)
    g=resize(take(G, X.flat - mmin),X.shape)
    b=resize(take(B, X.flat - mmin),X.shape)
    Y=mmconcat('d',r,g,b)
    return Y
#
# =====================================================================
#
#   mmgdtshow
#
# =====================================================================
def mmgdtshow(X, N=10):
    """
        - Purpose
            Apply an iso-line color table to a gray-scale image.
        - Synopsis
            Y = mmgdtshow(X, N=10)
        - Input
            X: Gray-scale (uint8 or uint16) image. Distance transform image.
            N: Default: 10. Number of iso-contours.
        - Output
            Y: Gray-scale (uint8 or uint16) or binary image.

    """
    from numpy import newaxis, ravel, ceil, zeros, ones, transpose, repeat, concatenate, arange, reshape, floor

    def apply_lut(img, lut):
        def lut_map(intens, lut=lut): return lut[intens]
        g = reshape(transpose(map(lut_map, ravel(img))), (3,img.shape[0],img.shape[1]))
        return g
    np = 1  # number of pixels by isoline
    if len(X.shape) == 1: X = X[newaxis,:]
    aux  = ravel(X)
    maxi, mini = max(aux), min(aux)
    d = int(ceil(256./N))
    m = zeros(256); m[0:256:d] = 1
    m = transpose([m,m,m])
    # lut gray
    gray = floor(arange(N)*255. / (N-1) + 0.5).astype('b')
    gray = repeat(gray, d)[0:256]
    gray = transpose([gray,gray,gray])
    # lut jet
    r = concatenate((range(126,0,-4),zeros(64),range(0,255,4),255*ones(64),range(255,128,-4)))
    g = concatenate((zeros(32),range(0,255,4),255*ones(64),range(255,0,-4),zeros(32)))
    b = 255 - r
    jet = transpose([r,g,b])
    # apply lut
    XX  = reshape(floor((aux-mini)*255. / maxi + 0.5).astype('b'), X.shape)
    lut = (1-m)*gray + m*jet
    Y = apply_lut(XX, lut)
    return Y
#
# =====================================================================
#
#   mmgshow
#
# =====================================================================
def mmgshow(X, X1=None, X2=None, X3=None, X4=None, X5=None, X6=None):
    """
        - Purpose
            Apply binary overlays as color layers on a binary or gray-scale
            image
        - Synopsis
            Y = mmgshow(X, X1=None, X2=None, X3=None, X4=None, X5=None,
            X6=None)
        - Input
            X:  Gray-scale (uint8 or uint16) or binary image.
            X1: Binary image. Default: None. Red overlay.
            X2: Binary image. Default: None. Green overlay.
            X3: Binary image. Default: None. Blue overlay.
            X4: Binary image. Default: None. Magenta overlay.
            X5: Binary image. Default: None. Yellow overlay.
            X6: Binary image. Default: None. Cyan overlay.
        - Output
            Y: Gray-scale (uint8 or uint16) or binary image.

    """

    if mmisbinary(X): X = mmgray(X,'uint8')
    r = X
    g = X
    b = X
    if X1 is not None: # red 1 0 0
      assert mmisbinary(X1),'X1 must be binary overlay'
      x1 = mmgray(X1,'uint8')
      r = mmunion(r,x1)
      g = mmintersec(g,mmneg(x1))
      b = mmintersec(b,mmneg(x1))
    if X2 is not None: # green 0 1 0
      assert mmisbinary(X2),'X2 must be binary overlay'
      x2 = mmgray(X2,'uint8')
      r = mmintersec(r,mmneg(x2))
      g = mmunion(g,x2)
      b = mmintersec(b,mmneg(x2))
    if X3 is not None: # blue 0 0 1
      assert mmisbinary(X3),'X3 must be binary overlay'
      x3 = mmgray(X3,'uint8')
      r = mmintersec(r,mmneg(x3))
      g = mmintersec(g,mmneg(x3))
      b = mmunion(b,x3)
    if X4 is not None: # magenta 1 0 1
      assert mmisbinary(X4),'X4 must be binary overlay'
      x4 = mmgray(X4,'uint8')
      r = mmunion(r,x4)
      g = mmintersec(g,mmneg(x4))
      b = mmunion(b,x4)
    if X5 is not None: # yellow 1 1 0
      assert mmisbinary(X5),'X5 must be binary overlay'
      x5 = mmgray(X5,'uint8')
      r = mmunion(r,x5)
      g = mmunion(g,x5)
      b = mmintersec(b,mmneg(x5))
    if X6 is not None: # cyan 0 1 1
      assert mmisbinary(X6),'X6 must be binary overlay'
      x6 = mmgray(X6,'uint8')
      r = mmintersec(r,mmneg(x6))
      g = mmunion(g,x6)
      b = mmunion(b,x6)
    return mmconcat('d',r,g,b)
    return Y
#
# =====================================================================
#
#   mmhistogram
#
# =====================================================================
def mmhistogram(f, option="uint16"):
    """
        - Purpose
            Find the histogram of the image f.
        - Synopsis
            h = mmhistogram(f, option="uint16")
        - Input
            f:      Gray-scale (uint8 or uint16) or binary image.
            option: String Default: "uint16". Values: "uint16" or "int32".
        - Output
            h: Gray-scale (uint8 or uint16) image. Histogram in a uint16 or
               an int32 vector.
        - Description
            Finds the histogram of the image f and returns the result in the
            vector h . For binary image the vector size is 2, for gray-scale
            uint8 and uint16 images, the vector size is the maximum pixel
            value plus one. h[0] gives the number of pixels with value 0.
        - Examples
            #
            #   example 1
            #
            f=to_uint8([0, 1, 1, 2, 2, 2, 5, 3, 5])
            h=mmhistogram(f)
            print h
            #
            #   example 2
            #
            f=mmreadgray('lenina.tif')
            mmshow(f)
            h=mmhistogram(f)
            mmplot([[h]],[['style', 'impulses']])
    """
    from numpy import searchsorted, sort, ravel, concatenate, product

    n = searchsorted(sort(ravel(f)), range(max(ravel(f))+1))
    n = concatenate([n, [product(f.shape)]])
    h = n[1:]-n[:-1]
    return h
#
# =====================================================================
#
#   mmlabel
#
# =====================================================================
def mmlabel(f, Bc=None):
    """
        - Purpose
            Label a binary image.
        - Synopsis
            y = mmlabel(f, Bc=None)
        - Input
            f:  Binary image.
            Bc: Structuring Element Default: None (3x3 elementary cross). (
                connectivity).
        - Output
            y: Image If number of labels is less than 65535, the data type
               is uint16, otherwise it is int32.
        - Description
            mmlabel creates the image y by labeling the connect components
            of a binary image f , according to the connectivity defined by
            the structuring element Bc . The background pixels (with value
            0) are not labeled. The maximum label value in the output image
            gives the number of its connected components.
        - Examples
            #
            #   example 1
            #
            f=mmbinary([
               [0,1,0,1,1],
               [1,0,0,1,0]])
            g=mmlabel(f)
            print g
            #
            #   example 2
            #
            f = mmreadgray('blob3.tif')
            g=mmlabel(f)
            nblobs=mmstats(g,'max')
            print nblobs
            mmshow(f)
            mmlblshow(g)
    """
    from numpy import allclose, ravel, nonzero, array
    if Bc is None: Bc = mmsecross()
    assert mmisbinary,'Can only label binary image'
    zero = mmsubm(f,f)               # zero image
    faux=f
    r = array(zero)
    label = 1
    y = mmgray( f,'uint16',0)        # zero image (output)
    while not allclose(faux,0):
        x=nonzero(ravel(faux))[0]      # get first unlabeled pixel
        fmark = array(zero)
        fmark.flat[x] = 1              # get the first unlabeled pixel
        r = mminfrec( fmark, faux, Bc) # detects all pixels connected to it
        faux = mmsubm( faux, r)        # remove them from faux
        r = mmgray( r,'uint16',label)  # label them with the value label
        y = mmunion( y, r)             # merge them with the labeled image
        label = label + 1
    return y
#
# =====================================================================
#
#   mmneg
#
# =====================================================================
def mmneg(f):
    """
        - Purpose
            Negate an image.
        - Synopsis
            y = mmneg(f)
        - Input
            f: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image.
        - Output
            y: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image.
        - Description
            mmneg returns an image y that is the negation (i.e., inverse or
            involution) of the image f . In the binary case, y is the
            complement of f .
        - Examples
            #
            #   example 1
            #
            f=to_uint8([255, 255, 0, 10, 20, 10, 0, 255, 255])
            print mmneg(f)
            print mmneg(to_uint8([0, 1]))
            print mmneg(int32([0, 1]))
            #
            #   example 2
            #
            a = mmreadgray('gear.tif')
            b = mmneg(a)
            mmshow(a)
            mmshow(b)
            #
            #   example 3
            #
            c = mmreadgray('astablet.tif')
            d = mmneg(c)
            mmshow(c)
            mmshow(d)
    """

    y = mmlimits(f)[0] + mmlimits(f)[1] - f
    y = y.astype(f.dtype)
    return y
#
# =====================================================================
#
#   mmthreshad
#
# =====================================================================
def mmthreshad(f, f1, f2=None):
    """
        - Purpose
            Threshold (adaptive)
        - Synopsis
            y = mmthreshad(f, f1, f2=None)
        - Input
            f:  Gray-scale (uint8 or uint16) image.
            f1: Gray-scale (uint8 or uint16) image. lower value
            f2: Gray-scale (uint8 or uint16) image. Default: None. upper
                value
        - Output
            y: Binary image.
        - Description
            mmthreshad creates the image y as the threshold of the image f
            by the images f1 and f2 . A pixel in y has the value 1 when the
            value of the corresponding pixel in f is between the values of
            the corresponding pixels in f1 and f2 .
        - Examples
            #
            a = mmreadgray('keyb.tif')
            mmshow(a)
            b = mmthreshad(a,to_uint8(10), to_uint8(50))
            mmshow(b)
            c = mmthreshad(a,238)
            mmshow(c)
    """

    if f2 is None: 
      y = mmbinary(f1 <= f)
    else:
      y = mmbinary((f1 <= f) & (f <= f2))
    return y
#
# =====================================================================
#
#   mmtoggle
#
# =====================================================================
def mmtoggle(f, f1, f2, OPTION="GRAY"):
    """
        - Purpose
            Image contrast enhancement or classification by the toggle
            operator.
        - Synopsis
            y = mmtoggle(f, f1, f2, OPTION="GRAY")
        - Input
            f:      Gray-scale (uint8 or uint16) image.
            f1:     Gray-scale (uint8 or uint16) image.
            f2:     Gray-scale (uint8 or uint16) image.
            OPTION: String Default: "GRAY". Values: 'BINARY' or 'GRAY'.
        - Output
            y: Image binary image if option is 'BINARY' or same type as f
        - Description
            mmtoggle creates the image y that is an enhancement or
            classification of the image f by the toggle operator, with
            parameters f1 and f2 . If the OPTION is 'GRAY', it performs an
            enhancement and, if the OPTION is 'BINARY', it performs a binary
            classification. In the enhancement, a pixel takes the value of
            the corresponding pixel in f1 or f2 , according to a minimum
            distance criterion from f to f1 or f to f2 . In the
            classification, the pixels in f nearest to f1 receive the value
            0 , while the ones nearest to f2 receive the value 1.
        - Examples
            #
            #   example 1
            #
            f = to_uint8([0,1,2,3,4,5,6])
            print f
            f1 = to_uint8([0,0,0,0,0,0,0])
            print f1
            f2 = to_uint8([6,6,6,6,6,6,6])
            print f2
            print mmtoggle(f,f1,f2)
            #
            #   example 2
            #
            a = mmreadgray('angiogr.tif')
            b = mmero(a,mmsedisk(2))
            c = mmdil(a,mmsedisk(2))
            d = mmtoggle(a,b,c)
            mmshow(a)
            mmshow(d)
            #
            #   example 3
            #
            e = mmreadgray('lenina.tif')
            f = mmero(e,mmsedisk(2))
            g = mmdil(e,mmsedisk(2))
            h = mmtoggle(e,f,g,'BINARY')
            mmshow(e)
            mmshow(h)
    """
    from string import upper

    y=mmbinary(mmsubm(f,f1),mmsubm(f2,f))
    if upper(OPTION) == 'GRAY':
        t=mmgray(y)
        y=mmunion(mmintersec(mmneg(t),f1),mmintersec(t,f2))
    return y
#
# =====================================================================
#
#   mmaddm
#
# =====================================================================
def mmaddm(f1, f2):
    """
        - Purpose
            Addition of two images, with saturation.
        - Synopsis
            y = mmaddm(f1, f2)
        - Input
            f1: Unsigned gray-scale (uint8 or uint16), signed (int32) or
                binary image.
            f2: Unsigned gray-scale (uint8 or uint16), signed (int32) or
                binary image. Or constant.
        - Output
            y: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image.
        - Description
            mmaddm creates the image y by pixelwise addition of images f1
            and f2 . When the addition of the values of two pixels saturates
            the image data type considered, the greatest value of this type
            is taken as the result of the addition.
        - Examples
            #
            #   example 1
            #
            f = to_uint8([255,   255,    0,   10,    0,   255,   250])
            g = to_uint8([ 0,    40,   80,   140,  250,    10,    30])
            y1 = mmaddm(f,g)
            print y1
            y2 = mmaddm(g, 100)
            print y2
            #
            #   example 2
            #
            a = mmreadgray('keyb.tif')
            b = mmaddm(a,128)
            mmshow(a)
            mmshow(b)
    """
    from numpy import array, minimum, maximum

    if type(f2) is array:
        assert f1.dtype == f2.dtype, 'Cannot have different datatypes:'
    y = maximum(minimum(f1.astype('d')+f2, mmlimits(f1)[1]),mmlimits(f1)[0])
    y = y.astype(f1.dtype)
    return y
#
# =====================================================================
#
#   mmareaclose
#
# =====================================================================
def mmareaclose(f, a, Bc=None):
    """
        - Purpose
            Area closing
        - Synopsis
            y = mmareaclose(f, a, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) or binary image.
            a:  Double non negative integer.
            Bc: Structuring Element Default: None (3x3 elementary cross). (
                connectivity).
        - Output
            y: Same type of f
        - Description
            mmareaclose removes any pore (i.e., background connected
            component) with area less than a of a binary image f . The
            connectivity is given by the structuring element Bc . This
            operator is generalized to gray-scale images by applying the
            binary operator successively on slices of f taken from higher
            threshold levels to lower threshold levels.
        - Examples
            #
            #   example 1
            #
            a=mmreadgray('form-1.tif')
            b=mmareaclose(a,400)
            mmshow(a)
            mmshow(b)
            #
            #   example 2
            #
            a=mmreadgray('n2538.tif')
            b=mmareaclose(a,400)
            mmshow(a)
            mmshow(b)
    """

    if Bc is None: Bc = mmsecross()
    y = mmneg(mmareaopen(mmneg(f),a,Bc))
    return y
#
# =====================================================================
#
#   mmareaopen
#
# =====================================================================
def mmareaopen(f, a, Bc=None):
    """
        - Purpose
            Area opening
        - Synopsis
            y = mmareaopen(f, a, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) or binary image.
            a:  Double non negative integer.
            Bc: Structuring Element Default: None (3x3 elementary cross). (
                connectivity).
        - Output
            y: Same type of f
        - Description
            mmareaopen removes any grain (i.e., connected component) with
            area less than a of a binary image f . The connectivity is given
            by the structuring element Bc . This operator is generalized to
            gray-scale images by applying the binary operator successively
            on slices of f taken from higher threshold levels to lower
            threshold levels.
        - Examples
            #
            #   example 1
            #
            f=mmbinary(to_uint8([
             [1, 1, 0, 0, 0, 0, 1],
             [1, 0, 1, 1, 1, 0, 1],
             [0, 0, 0, 0, 1, 0, 0]]))
            y=mmareaopen(f,4,mmsecross())
            print y
            #
            #   example 2
            #
            f=to_uint8([
               [10,   11,   0,    0,   0,   0,  20],
               [10,    0,   5,    8,   9,   0,  15],
               [10,    0,   0,    0,  10,   0,   0]])
            y=mmareaopen(f,4,mmsecross())
            print y
            #
            #   example 3
            #
            a=mmreadgray('form-1.tif');
            b=mmareaopen(a,500);
            mmshow(a);
            mmshow(b);
            #
            #   example 4
            #
            a=mmreadgray('bloodcells.tif');
            b=mmareaopen(a,500);
            mmshow(a);
            mmshow(b);
    """

    if Bc is None: Bc = mmsecross()
    if mmisbinary(f):
      fr = mmlabel(f,Bc)      # binary area open, use area measurement
      g = mmblob(fr,'area')
      y = mmthreshad(g,a)
    else:
      y = mmintersec(f,0)
      zero = mmbinary(y)
      k1 = mmstats(f,'min')
      k2 = mmstats(f,'max')
      for k in range(k1,k2+1):   # gray-scale, use thresholding decomposition
        fk = mmthreshad(f,k)
        fo = mmareaopen(fk,a,Bc)
        if mmisequal(fo,zero):
          break
        y = mmunion(y, mmgray(fo,mmdatatype(f),k))
    return y
#
# =====================================================================
#
#   mmflood
#
# =====================================================================
def mmflood(fin, T, option, Bc=None):
    """
        - Purpose
            Flooding filter- h,v,a-basin and dynamics (depth, area, volume)
        - Synopsis
            y = mmflood(fin, T, option, Bc=None)
        - Input
            fin:    Gray-scale (uint8 or uint16) image.
            T:      Criterion value. If T==-1, then the dynamics is
                    determined, not the flooding at this criterion. This was
                    selected just to use the same algoritm to compute two
                    completely distinct functions.
            option: String Default: "". criterion: 'AREA', 'VOLUME', 'H'.
            Bc:     Structuring Element Default: None (3x3 elementary
                    cross). Connectivity.
        - Output
            y: Gray-scale (uint8 or uint16) image.
        - Description
            This is a flooding algorithm. It is the basis to implement many
            topological functions. It is a connected filter that floods an
            image following some topological criteria: area, volume, depth.
            These filters are equivalent to area-close, volume-basin or
            h-basin, respectively. This code may be difficult to understand
            because of its many options. Basically, when t is negative, the
            generalized dynamics: area, volume, h is computed. When the
            flooding is computed, every time a new level in the flooding
            happens, a test is made to verify if the criterion has reached.
            This is used to set the value to that height. This value image
            will be used later for sup-reconstruction (flooding) at that
            particular level. This test happens in the raising of the water
            and in the merging of basins.

    """

    if Bc is None: Bc = mmsecross()
    print 'Not implemented yet'
    return None
    return y
#
# =====================================================================
#
#   mmasf
#
# =====================================================================
def mmasf(f, SEQ="OC", b=None, n=1):
    """
        - Purpose
            Alternating Sequential Filtering
        - Synopsis
            y = mmasf(f, SEQ="OC", b=None, n=1)
        - Input
            f:   Gray-scale (uint8 or uint16) or binary image.
            SEQ: String Default: "OC". 'OC', 'CO', 'OCO', 'COC'.
            b:   Structuring Element Default: None (3x3 elementary cross).
            n:   Non-negative integer. Default: 1. (number of iterations).
        - Output
            y: Image
        - Description
            mmasf creates the image y by filtering the image f by n
            iterations of the close and open alternating sequential filter
            characterized by the structuring element b . The sequence of
            opening and closing is controlled by the parameter SEQ . 'OC'
            performs opening after closing, 'CO' performs closing after
            opening, 'OCO' performs opening after closing after opening, and
            'COC' performs closing after opening after closing.
        - Examples
            #
            #   example 1
            #
            f=mmreadgray('gear.tif')
            g=mmasf(f,'oc',mmsecross(),2)
            mmshow(f)
            mmshow(g)
            #
            #   example 2
            #
            f=mmreadgray('fabric.tif')
            g=mmasf(f,'oc',mmsecross(),3)
            mmshow(f)
            mmshow(g)
    """
    from string import upper
    if b is None: b = mmsecross()
    SEQ=upper(SEQ)
    y = f
    if SEQ == 'OC':
        for i in range(1,n+1):
            nb = mmsesum(b,i)
            y = mmopen(mmclose(y,nb),nb)
    elif SEQ == 'CO':
        for i in range(1,n+1):
            nb = mmsesum(b,i)
            y = mmclose(mmopen(y,nb),nb)
    elif SEQ == 'OCO':
        for i in range(1,n+1):
            nb = mmsesum(b,i)
            y = mmopen(mmclose(mmopen(y,nb),nb),nb)
    elif SEQ == 'COC':
        for i in range(1,n+1):
            nb = mmsesum(b,i)
            y = mmclose(mmopen(mmclose(y,nb),nb),nb)
    return y
#
# =====================================================================
#
#   mmasfrec
#
# =====================================================================
def mmasfrec(f, SEQ="OC", b=None, bc=None, n=1):
    """
        - Purpose
            Reconstructive Alternating Sequential Filtering
        - Synopsis
            y = mmasfrec(f, SEQ="OC", b=None, bc=None, n=1)
        - Input
            f:   Gray-scale (uint8 or uint16) or binary image.
            SEQ: String Default: "OC". Values: "OC" or "CO".
            b:   Structuring Element Default: None (3x3 elementary cross).
            bc:  Structuring Element Default: None (3x3 elementary cross).
            n:   Non-negative integer. Default: 1. (number of iterations).
        - Output
            y: Same type of f
        - Description
            mmasf creates the image y by filtering the image f by n
            iterations of the close by reconstruction and open by
            reconstruction alternating sequential filter characterized by
            the structuring element b . The structure element bc is used in
            the reconstruction. The sequence of opening and closing is
            controlled by the parameter SEQ . 'OC' performs opening after
            closing, and 'CO' performs closing after opening.
        - Examples
            #
            f=mmreadgray('fabric.tif')
            g=mmasfrec(f,'oc',mmsecross(),mmsecross(),3)
            mmshow(f)
            mmshow(g)
    """
    from string import upper
    if b is None: b = mmsecross()
    if bc is None: bc = mmsecross()
    SEQ = upper(SEQ)
    y = f
    if SEQ == 'OC':
        for i in range(1,n+1):
            nb = mmsesum(b,i)
            y = mmcloserec(y,nb,bc)
            y = mmopenrec(y,nb,bc)
    elif SEQ == 'CO':
        for i in range(1,n+1):
            nb = mmsesum(b,i)
            y = mmopenrec(y,nb,bc)
            y = mmcloserec(y,nb,bc)
    else:
        assert 0,'Only accepts OC or CO for SEQ parameter'
    return y
#
# =====================================================================
#
#   mmbinary
#
# =====================================================================
def mmbinary(f, k=1):
    """
        - Purpose
            Convert a gray-scale image into a binary image
        - Synopsis
            y = mmbinary(f, k1=1)
        - Input
            f:  Unsigned gray-scale (uint8 or uint16), signed (int32) or
                binary image.
            k1: Double Default: 1. Threshold value.
        - Output
            y: Binary image.
        - Description
            mmbinary converts a gray-scale image f into a binary image y by
            a threshold rule. A pixel in y has the value 1 if and only if
            the corresponding pixel in f has a value greater or equal k1 .
        - Examples
            #
            #   example 1
            #
            a = array([0, 1, 2, 3, 4])
            b=mmbinary(a)
            print b
            #
            #   example 2
            #
            a=mmreadgray('mm3.tif')
            b=mmbinary(a,82)
            mmshow(a)
            mmshow(b)
    """
    from numpy import asarray
    f=asarray(f)
    return (f >= k)
#
# =====================================================================
#
#   mmblob
#
# =====================================================================
def mmblob(fr, measurement, option="image"):
    """
        - Purpose
            Blob measurements from a labeled image.
        - Synopsis
            y = mmblob(fr, measurement, option="image")
        - Input
            fr:          Gray-scale (uint8 or uint16) image. Labeled image.
            measurement: String Default: "". Choice from 'AREA', 'CENTROID',
                         or 'BOUNDINGBOX'.
            option:      String Default: "image". Output format: 'image':
                         results as a binary image; 'data': results a column
                         vector of measurements (double).
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            Take measurements from the labeled image fr . The measurements
            are: area, centroid, or bounding rectangle. The parameter option
            controls the output format: 'IMAGE': the result is an image;
            'DATA': the result is a double column vector with the
            measurement for each blob. The region with label zero is not
            measured as it is normally the background. The measurement of
            region with label 1 appears at the first row of the output.
        - Examples
            #
            #   example 1
            #
            fr=to_uint8([
               [1,1,1,0,0,0],
               [1,1,1,0,0,2],
               [1,1,1,0,2,2]])
            f_area=mmblob(fr,'area')
            print f_area
            f_cent=mmblob(fr,'centroid')
            print f_cent
            f_bb=mmblob(fr,'boundingbox')
            print f_bb
            d_area=mmblob(fr,'area','data')
            print d_area
            d_cent=mmblob(fr,'centroid','data')
            print d_cent
            d_bb=mmblob(fr,'boundingbox','data')
            print d_bb
            #
            #   example 2
            #
            f=mmreadgray('blob3.tif')
            fr=mmlabel(f)
            g=mmblob(fr,'area')
            mmshow(f)
            mmshow(g)
            #
            #   example 3
            #
            f=mmreadgray('blob3.tif')
            fr=mmlabel(f)
            centr=mmblob(fr,'centroid')
            mmshow(f,mmdil(centr))
            #
            #   example 4
            #
            f=mmreadgray('blob3.tif')
            fr=mmlabel(f)
            box=mmblob(fr,'boundingbox')
            mmshow(f,box)
    """
    from numpy import newaxis, ravel, zeros, sum, nonzero, sometrue, array
    from string import upper

    measurement = upper(measurement)
    option      = upper(option)
    if len(fr.shape) == 1: fr = fr[newaxis,:]
    n = max(ravel(fr))
    if option == 'DATA': y = []
    else               : y = zeros(fr.shape)
    if measurement == 'AREA':
        for i in range(1,n+1):
            aux  = fr==i
            area = sum(ravel(aux))
            if option == 'DATA': y.append(area)
            else               : y = y + area*aux
    elif measurement == 'CENTROID':
        for i in range(1,n+1):
            aux  = fr==i
            ind  = nonzero(ravel(aux))
            indx = ind / fr.shape[1]
            indy = ind % fr.shape[1]
            centroid = [sum(indx)/len(ind), sum(indy)/len(ind)]
            if option == 'DATA': y.append([centroid[1],centroid[0]])
            else               : y[centroid] = 1
    elif measurement == 'BOUNDINGBOX':
        for i in range(1,n+1):
            aux = fr==i
            aux1, aux2 = sometrue(aux,0), sometrue(aux,1)
            col , row  = nonzero(aux1)  , nonzero(aux2)
            if option == 'DATA': y.append([col[0],row[0],col[-1],row[-1]])
            else:
                y[row[0]:row[-1],col[0] ] = 1
                y[row[0]:row[-1],col[-1]] = 1
                y[row[0], col[0]:col[-1]] = 1
                y[row[-1],col[0]:col[-1]] = 1
    else:
        print "Measurement option should be 'AREA','CENTROID', or 'BOUNDINGBOX'."
    if option == 'DATA':
        y = array(y)
        if len(y.shape) == 1: y = y[:,newaxis]
    return y
#
# =====================================================================
#
#   mmcbisector
#
# =====================================================================
def mmcbisector(f, B, n):
    """
        - Purpose
            N-Conditional bisector.
        - Synopsis
            y = mmcbisector(f, B, n)
        - Input
            f: Binary image.
            B: Structuring Element
            n: positive integer ( filtering rate)
        - Output
            y: Binary image.
        - Description
            mmcbisector creates the binary image y by performing a filtering
            of the morphological skeleton of the binary image f , relative
            to the structuring element B . The strength of this filtering is
            controlled by the parameter n. Particularly, if n=0 , y is the
            morphological skeleton of f itself.
        - Examples
            #
            a=mmreadgray('blob2.tif')
            b=mmcbisector(a,mmsebox(),1)
            c=mmcbisector(a,mmsebox(),3)
            d=mmcbisector(a,mmsebox(),10)
            mmshow(a,b)
            mmshow(a,c)
            mmshow(a,d)
    """

    y = mmintersec(f,0)
    for i in range(n):
        nb = mmsesum(B,i)
        nbp = mmsesum(B,i+1)
        f1 = mmero(f,nbp)
        f2 = mmcdil(f1,f,B,n)
        f3 = mmsubm(mmero(f,nb),f2)
        y  = mmunion(y,f3)
    return y
#
# =====================================================================
#
#   mmcdil
#
# =====================================================================
def mmcdil(f, g, b=None, n=1):
    """
        - Purpose
            Dilate an image conditionally.
        - Synopsis
            y = mmcdil(f, g, b=None, n=1)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            g: Gray-scale (uint8 or uint16) or binary image. Conditioning
               image.
            b: Structuring Element Default: None (3x3 elementary cross).
            n: Non-negative integer. Default: 1. (number of iterations).
        - Output
            y: Image
        - Description
            mmcdil creates the image y by dilating the image f by the
            structuring element b conditionally to the image g . This
            operator may be applied recursively n times.
        - Examples
            #
            #   example 1
            #
            f = mmbinary(to_uint8([[1, 0, 0, 0, 0, 0, 0],\
                [0, 0, 0, 0, 0, 0, 0],\
                [0, 0, 0, 0, 1, 0, 0,]]))
            g = mmbinary(to_uint8([[1, 1, 1, 0, 0, 1, 1],\
                [1, 0, 1, 1, 1, 0, 0],\
                [0, 0, 0, 0, 1, 0, 0]]));
            y1=mmcdil(f,g,mmsecross())
            y2=mmcdil(f,g,mmsecross(),3)
            #
            #   example 2
            #
            f = to_uint8([\
                [   0,    0,   0,   80,   0,   0],\
                [   0,    0,   0,    0,   0,   0],\
                [  10,   10,   0,  255,   0,   0]])
            g = to_uint8([\
                [   0,    1,   2,   50,   4,   5],\
                [   2,    3,   4,    0,   0,   0],\
                [  12,  255,  14,   15,  16,  17]])
            y1=mmcdil(f,g,mmsecross())
            y2=mmcdil(f,g,mmsecross(),3)
            #
            #   example 3
            #
            g=mmreadgray('pcb1bin.tif')
            f=mmframe(g,5,5)
            y5=mmcdil(f,g,mmsecross(),5)
            y25=mmcdil(f,g,mmsecross(),25)
            mmshow(g)
            mmshow(g,f)
            mmshow(g,y5)
            mmshow(g,y25)
            #
            #   example 4
            #
            g=mmneg(mmreadgray('n2538.tif'))
            f=mmintersec(g,0)
            f=mmdraw(f,'LINE:40,30,60,30:END')
            y1=mmcdil(f,g,mmsebox())
            y30=mmcdil(f,g,mmsebox(),30)
            mmshow(g)
            mmshow(f)
            mmshow(y1)
            mmshow(y30)
    """

    if b is None: b = mmsecross()
    y = mmintersec(f,g)
    for i in range(n):
        aux = y
        y = mmintersec(mmdil(y,b),g)
        if mmisequal(y,aux): break
    return y
#
# =====================================================================
#
#   mmcero
#
# =====================================================================
def mmcero(f, g, b=None, n=1):
    """
        - Purpose
            Erode an image conditionally.
        - Synopsis
            y = mmcero(f, g, b=None, n=1)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            g: Gray-scale (uint8 or uint16) or binary image. Conditioning
               image.
            b: Structuring Element Default: None (3x3 elementary cross).
            n: Non-negative integer. Default: 1. (number of iterations).
        - Output
            y: Image
        - Description
            mmcero creates the image y by eroding the image f by the
            structuring element b conditionally to g . This operator may be
            applied recursively n times.
        - Examples
            #
            f = mmneg(mmtext('hello'))
            mmshow(f)
            g = mmdil(f,mmseline(7,90))
            mmshow(g)
            a1=mmcero(g,f,mmsebox())
            mmshow(a1)
            a13=mmcero(a1,f,mmsebox(),13)
            mmshow(a13)
    """

    if b is None: b = mmsecross()
    y = mmunion(f,g)
    for i in range(n):
        aux = y
        y = mmunion(mmero(y,b),g)
        if mmisequal(y,aux): break
    return y
#
# =====================================================================
#
#   mmclose
#
# =====================================================================
def mmclose(f, b=None):
    """
        - Purpose
            Morphological closing.
        - Synopsis
            y = mmclose(f, b=None)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            b: Structuring Element Default: None (3x3 elementary cross).
        - Output
            y: Image
        - Description
            mmclose creates the image y by the morphological closing of the
            image f by the structuring element b . In the binary case, the
            closing by a structuring element B may be interpreted as the
            intersection of all the binary images that contain the image f
            and have a hole equal to a translation of B . In the gray-scale
            case, there is a similar interpretation taking the functions
            umbra.
        - Examples
            #
            #   example 1
            #
            f=mmreadgray('blob.tif')
            bimg=mmreadgray('blob1.tif')
            b=mmimg2se(bimg)
            mmshow(f)
            mmshow(mmclose(f,b))
            mmshow(mmclose(f,b),mmgradm(f))
            #
            #   example 2
            #
            f = mmreadgray('form-1.tif')
            mmshow(f)
            y = mmclose(f,mmsedisk(4))
            mmshow(y)
            #
            #   example 3
            #
            f = mmreadgray('n2538.tif')
            mmshow(f)
            y = mmclose(f,mmsedisk(3))
            mmshow(y)
    """

    if b is None: b = mmsecross()
    y = mmero(mmdil(f,b),b)
    return y
#
# =====================================================================
#
#   mmcloserec
#
# =====================================================================
def mmcloserec(f, bdil=None, bc=None):
    """
        - Purpose
            Closing by reconstruction.
        - Synopsis
            y = mmcloserec(f, bdil=None, bc=None)
        - Input
            f:    Gray-scale (uint8 or uint16) or binary image.
            bdil: Structuring Element Default: None (3x3 elementary cross).
                  (dilation).
            bc:   Structuring Element Default: None (3x3 elementary cross).
                  ( connectivity).
        - Output
            y: Same type of f .
        - Description
            mmcloserec creates the image y by a sup-reconstruction ( with
            the connectivity defined by the structuring element bc ) of the
            image f from its dilation by bdil .
        - Examples
            #
            a = mmreadgray('danaus.tif')
            mmshow(a)
            b = mmcloserec(a,mmsebox(4))
            mmshow(b)
    """

    if bdil is None: bdil = mmsecross()
    if bc is None: bc = mmsecross()
    y = mmsuprec(mmdil(f,bdil),f,bc)
    return y
#
# =====================================================================
#
#   mmcloserecth
#
# =====================================================================
def mmcloserecth(f, bdil=None, bc=None):
    """
        - Purpose
            Close-by-Reconstruction Top-Hat.
        - Synopsis
            y = mmcloserecth(f, bdil=None, bc=None)
        - Input
            f:    Gray-scale (uint8 or uint16) or binary image.
            bdil: Structuring Element Default: None (3x3 elementary cross).
                  (dilation)
            bc:   Structuring Element Default: None (3x3 elementary cross).
                  ( connectivity)
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmcloserecth creates the image y by subtracting the image f of
            its closing by reconstruction, defined by the structuring
            elements bc and bdil .
        - Examples
            #
            a = mmreadgray('danaus.tif')
            mmshow(a)
            b = mmcloserecth(a,mmsebox(4))
            mmshow(b)
    """

    if bdil is None: bdil = mmsecross()
    if bc is None: bc = mmsecross()
    y = mmsubm(mmcloserec(f,bdil,bc), f)
    return y
#
# =====================================================================
#
#   mmcloseth
#
# =====================================================================
def mmcloseth(f, b=None):
    """
        - Purpose
            Closing Top Hat.
        - Synopsis
            y = mmcloseth(f, b=None)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            b: Structuring Element Default: None (3x3 elementary cross).
        - Output
            y: Gray-scale (uint8 or uint16) or binary image. (Same type of f
               ).
        - Description
            mmcloseth creates the image y by subtracting the image f of its
            morphological closing by the structuring element b .
        - Examples
            #
            a = mmreadgray('danaus.tif')
            mmshow(a)
            b = mmcloseth(a,mmsebox(5))
            mmshow(b)
    """

    if b is None: b = mmsecross()
    y = mmsubm( mmclose(f,b), f)
    return y
#
# =====================================================================
#
#   mmcmp
#
# =====================================================================
def mmcmp(f1, oper, f2, oper1=None, f3=None):
    """
        - Purpose
            Compare two images pixelwisely.
        - Synopsis
            y = mmcmp(f1, oper, f2, oper1=None, f3=None)
        - Input
            f1:    Gray-scale (uint8 or uint16) or binary image.
            oper:  String Default: "". relationship from: '==', '~=',
                   '<','<=', '>', '>='.
            f2:    Gray-scale (uint8 or uint16) or binary image.
            oper1: String Default: None. relationship from: '==', '~=',
                   '<','<=', '>', '>='.
            f3:    Gray-scale (uint8 or uint16) or binary image. Default:
                   None.
        - Output
            y: Binary image.
        - Description
            Apply the relation oper to each pixel of images f1 and f2 , the
            result is a binary image with the same size. Optionally, it is
            possible to make the comparison among three image. It is
            possible to use a constant value in place of any image, in this
            case the constant is treated as an image of the same size as the
            others with all pixels with the value of the constant.
        - Examples
            #
            #   example 1
            #
            print mmcmp(to_uint8([1, 2, 3]),'<', to_uint8(2))
            print mmcmp(to_uint8([1, 2, 3]),'<', to_uint8([0, 2, 4]))
            print mmcmp(to_uint8([1, 2, 3]),'==', to_uint8([1, 1, 3]))
            #
            #   example 2
            #
            f=mmreadgray('keyb.tif')
            fbin=mmcmp(to_uint8(10), '<', f, '<', to_uint8(50))
            mmshow(f)
            mmshow(fbin)
    """

    if   oper == '==':    y = (f1==f2)
    elif oper == '~=':    y = (f1!=f2)
    elif oper == '<=':    y = (f1<=f2)
    elif oper == '>=':    y = (f1>=f2)
    elif oper == '>':     y = (f1> f2)
    elif oper == '<':     y = (f1< f2)
    else:
        assert 0, 'oper must be one of: ==, ~=, >, >=, <, <=, it was:'+oper
    if oper1 != None:
        if   oper1 == '==':     y = mmintersec(y, f2==f3)
        elif oper1 == '~=':     y = mmintersec(y, f2!=f3)
        elif oper1 == '<=':     y = mmintersec(y, f2<=f3)
        elif oper1 == '>=':     y = mmintersec(y, f2>=f3)
        elif oper1 == '>':      y = mmintersec(y, f2> f3)
        elif oper1 == '<':      y = mmintersec(y, f2< f3)
        else:
            assert 0, 'oper1 must be one of: ==, ~=, >, >=, <, <=, it was:'+oper1

    y = mmbinary(y)
    return y
#
# =====================================================================
#
#   mmcthick
#
# =====================================================================
def mmcthick(f, g, Iab=None, n=-1, theta=45, DIRECTION="CLOCKWISE"):
    """
        - Purpose
            Image transformation by conditional thickening.
        - Synopsis
            y = mmcthick(f, g, Iab=None, n=-1, theta=45,
            DIRECTION="CLOCKWISE")
        - Input
            f:         Binary image.
            g:         Binary image.
            Iab:       Interval Default: None (mmhomothick).
            n:         Non-negative integer. Default: -1. Number of
                       iterations.
            theta:     Double Default: 45. Degrees of rotation: 45, 90, or
                       180.
            DIRECTION: String Default: "CLOCKWISE". 'CLOCKWISE' or
                       'ANTI-CLOCKWISE'.
        - Output
            y: Binary image.
        - Description
            mmcthick creates the binary image y by performing a thickening
            of the binary image f conditioned to the binary image g . The
            number of iterations of the conditional thickening is n and in
            each iteration the thickening is characterized by rotations of
            theta of the interval Iab .
        - Examples
            #
            #   example 1
            #
            f=mmreadgray('blob2.tif')
            mmshow(f)
            t=mmse2hmt(mmbinary([[0,0,0],[0,0,1],[1,1,1]]),
                                      mmbinary([[0,0,0],[0,1,0],[0,0,0]]))
            print mmintershow(t)
            f1=mmthick(f,t,40); # The thickening makes the image border grow
            mmshow(f1)
            #
            #   example 2
            #
            f2=mmcthick(f,mmneg(mmframe(f)),t,40) # conditioning to inner pixels
            fn=mmcthick(f,mmneg(mmframe(f)),t) #pseudo convex hull
            mmshow(f2)
            mmshow(fn,f)
    """
    from numpy import product
    from string import upper
    if Iab is None: Iab = mmhomothick()
    DIRECTION = upper(DIRECTION)            
    assert mmisbinary(f),'f must be binary image'
    if n == -1: n = product(f.shape)
    y = f
    old = y
    for i in range(n):
        for t in range(0,360,theta):
            sup = mmsupgen( y, mminterot(Iab, t, DIRECTION))
            y = mmintersec(mmunion( y, sup),g)
        if mmisequal(old,y): break
        old = y
    return y
#
# =====================================================================
#
#   mmcthin
#
# =====================================================================
def mmcthin(f, g, Iab=None, n=-1, theta=45, DIRECTION="CLOCKWISE"):
    """
        - Purpose
            Image transformation by conditional thinning.
        - Synopsis
            y = mmcthin(f, g, Iab=None, n=-1, theta=45,
            DIRECTION="CLOCKWISE")
        - Input
            f:         Binary image.
            g:         Binary image.
            Iab:       Interval Default: None (mmhomothin).
            n:         Non-negative integer. Default: -1. Number of
                       iterations.
            theta:     Double Default: 45. Degrees of rotations: 45, 90, or
                       180.
            DIRECTION: String Default: "CLOCKWISE". 'CLOCKWISE' or '
                       ANTI-CLOCKWISE'.
        - Output
            y: Binary image.
        - Description
            mmcthin creates the binary image y by performing a thinning of
            the binary image f conditioned to the binary image g . The
            number of iterations of the conditional thinning is n and in
            each iteration the thinning is characterized by rotations of
            theta of the interval Iab .

    """
    from numpy import product
    from string import upper
    if Iab is None: Iab = mmhomothin()
    DIRECTION = upper(DIRECTION)            
    assert mmisbinary(f),'f must be binary image'
    if n == -1: n = product(f.shape)
    y = f
    old = y
    for i in range(n):
        for t in range(0,360,theta):
            sup = mmsupgen( y, mminterot(Iab, t, DIRECTION))
            y = mmunion(mmsubm( y, sup),g)
        if mmisequal(old,y): break
        old = y
    return y
#
# =====================================================================
#
#   mmcwatershed
#
# =====================================================================
def mmcwatershed(f, g, Bc=None, LINEREG="LINES"):
    """
        - Purpose
            Detection of watershed from markers.
        - Synopsis
            Y = mmcwatershed(f, g, Bc=None, LINEREG="LINES")
        - Input
            f:       Gray-scale (uint8 or uint16) image.
            g:       Gray-scale (uint8 or uint16) or binary image. marker
                     image: binary or labeled.
            Bc:      Structuring Element Default: None (3x3 elementary
                     cross). (watershed connectivity)
            LINEREG: String Default: "LINES". 'LINES' or ' REGIONS'.
        - Output
            Y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmcwatershed creates the image y by detecting the domain of the
            catchment basins of f indicated by the marker image g ,
            according to the connectivity defined by Bc . According to the
            flag LINEREG y will be a labeled image of the catchment basins
            domain or just a binary image that presents the watershed lines.
            To know more about watershed and watershed from markers, see
            BeucMeye:93 . The implementation of this function is based on
            LotuFalc:00 . WARNING: There is a common mistake related to the
            marker image g . If this image contains only zeros and ones, but
            it is not a binary image, the result will be an image with all
            ones. If the marker image is binary, you have to set this
            explicitly using the logical function.
        - Examples
            #
            #   example 1
            #
            a = to_uint8([\
                [10,   10,   10,   10,   10,   10,   10],\
                [10,    9,    6,   18,    6,    5,   10],\
                [10,    9,    6,   18,    6,    8,   10],\
                [10,    9,    9,   15,    9,    9,   10],\
                [10,    9,    9,   15,   12,   10,   10],\
                [10,   10,   10,   10,   10,   10,   10]])
            b = mmcmp(a,'==',to_uint8(6))
            print mmcwatershed(a,b)
            print mmcwatershed(a,b,mmsecross(),'REGIONS')
            #
            #   example 2
            #
            f=mmreadgray('astablet.tif')
            grad=mmgradm(f)
            mark=mmregmin(mmhmin(grad,17))
            w=mmcwatershed(grad,mark)
            mmshow(grad)
            mmshow(mark)
            mmshow(w)
    """
    from numpy import ones, zeros, nonzero, array, put, take, argmin, transpose, compress, concatenate
    if Bc is None: Bc = mmsecross()
    return g
    print 'starting'
    withline = (LINEREG == 'LINES')
    if mmis(g,'binary'):
        g = mmlabel(g,Bc)
    print 'before 1. mmpad4n'
    status = mmpad4n(to_uint8(zeros(f.shape)),Bc, 3)
    f = mmpad4n( f,Bc,0)                 #pad input image
    print 'before 2. mmpad4n'
    y = mmpad4n( g,Bc,0)                  # pad marker image with 0
    if withline:
        y1 = mmintersec(mmbinary(y), 0)
    costM = mmlimits(f)[1] * ones(f.shape)  # cummulative cost function image
    mi = nonzero(mmgradm(y,mmsebox(0),Bc).flat)  # 1D index of internal contour of marker
    print 'before put costM'
    put(costM.flat,mi, 0)
    HQueue=transpose([mi, take(costM.flat, mi)])       # init hierarquical queue: index,value
    print 'before mmse2list0'
    Bi=mmse2list0(f,Bc)                # get 1D displacement neighborhood pixels
    x,v = mmmat2set(Bc)
    while HQueue:
        print 'Hq=',HQueue
        i = argmin(HQueue[:,1])           # i is the index of minimum value
        print 'imin=',i
        pi = HQueue[i,0]
        print 'pi=',pi
        ii = ones(HQueue.shape[0])
        ii[i] = 0
        print 'ii=',ii
        HQueue = transpose(array([compress(ii,HQueue[:,0]),
                                  compress(ii,HQueue[:,1])])) # remove this pixel from queue
        print 'H=',HQueue
        put(status.flat, pi, 1)          # make it a permanent label
        for qi in pi+Bi :                # for each neighbor of pi
            if (status.flat[qi] != 3):          # not image border
                if (status.flat[qi] != 1):        # if not permanent
                    cost_M = max(costM.flat[pi], f.flat[qi])
                    if cost_M < costM.flat[qi]:
                        print 'qi=',qi
                        costM.flat[qi] = cost_M
                        y.flat[qi] = y.flat[pi]                  # propagate the label
                        aux = zeros(array(HQueue.shape) + [1,0])
                        aux[:-1,:] = HQueue
                        aux[-1,:]=[qi, cost_M]
                        HQueue = aux # insert pixel in the queue
                        print 'insert H=',HQueue
                elif (withline        and
                     (y.flat[qi] != y.flat[pi]) and
                     (y1.flat[pi] == 0)    and
                     (y1.flat[qi] == 0)     ):
                    y1.flat[pi] = 1
    if withline:
        Y = y1
    else:
        Y = y
    return Y
#
# =====================================================================
#
#   mmdil
#
# =====================================================================
def mmdil(f, b=None):
    """
        - Purpose
            Dilate an image by a structuring element.
        - Synopsis
            y = mmdil(f, b=None)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            b: Structuring Element Default: None (3x3 elementary cross).
        - Output
            y: Image
        - Description
            mmdil performs the dilation of image f by the structuring
            element b . Dilation is a neighbourhood operator that compares
            locally b with f , according to an intersection rule. Since
            Dilation is a fundamental operator to the construction of all
            other morphological operators, it is also called an elementary
            operator of Mathematical Morphology. When f is a gray-scale
            image, b may be a flat or non-flat structuring element.
        - Examples
            #
            #   example 1
            #
            f=mmbinary([
               [0, 0, 0, 0, 0, 0, 1],
               [0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0]])
            b=mmbinary([1, 1, 0])
            mmdil(f,b)
            f=to_uint8([
               [ 0,   1,  2, 50,  4,  5],
               [ 2,   3,  4,  0,  0,  0],
               [12, 255, 14, 15, 16, 17]])
            mmdil(f,b)
            #
            #   example 2
            #
            f=mmbinary(mmreadgray('blob.tif'))
            bimg=mmbinary(mmreadgray('blob1.tif'))
            b=mmimg2se(bimg)
            mmshow(f)
            mmshow(mmdil(f,b))
            mmshow(mmdil(f,b),mmgradm(f))
            #
            #   example 3
            #
            f=mmreadgray('pcb_gray.tif')
            b=mmsedisk(5)
            mmshow(f)
            mmshow(mmdil(f,b))
    """
    from numpy import maximum, newaxis, ones
    if b is None: b = mmsecross()
    if len(f.shape) == 1: f = f[newaxis,:]
    h,w = f.shape
    x,v = mmmat2set(b)
    if len(x)==0:
        y = (ones((h,w)) * mmlimits(f)[0]).astype(f.dtype)
    else:
        if mmisbinary(v):
            v = mmintersec(mmgray(v,'int32'),0)
        mh,mw = max(abs(x)[:,0]),max(abs(x)[:,1])
        y = (ones((h+2*mh,w+2*mw)) * mmlimits(f)[0]).astype(f.dtype)
        for i in range(x.shape[0]):
            if v[i] > -2147483647:
                y[mh+x[i,0]:mh+x[i,0]+h, mw+x[i,1]:mw+x[i,1]+w] = maximum(
                    y[mh+x[i,0]:mh+x[i,0]+h, mw+x[i,1]:mw+x[i,1]+w], mmadd4dil(f,v[i]))
        y = y[mh:mh+h, mw:mw+w]
    return y
#
# =====================================================================
#
#   mmdrawv
#
# =====================================================================
def mmdrawv(f, data, value, GEOM):
    """
        - Purpose
            Superpose points, rectangles and lines on an image.
        - Synopsis
            y = mmdrawv(f, data, value, GEOM)
        - Input
            f:     Gray-scale (uint8 or uint16) or binary image.
            data:  Gray-scale (uint8 or uint16) or binary image. vector of
                   points. Each row gives information regarding a
                   geometrical primitive. The interpretation of this data is
                   dependent on the parameter GEOM. The line drawing
                   algorithm is not invariant to image transposition.
            value: Gray-scale (uint8 or uint16) or binary image. pixel
                   gray-scale value associated to each point in parameter
                   data. It can be a column vector of values or a single
                   value.
            GEOM:  String Default: "". geometrical figure. One of
                   'point','line', 'rect', or 'frect' for drawing points,
                   lines, rectangles or filled rectangles respectively.
        - Output
            y: Gray-scale (uint8 or uint16) or binary image. y has the same
               type of f .
        - Description
            mmdrawv creates the image y by a superposition of points,
            rectangles and lines of gray-level k1 on the image f . The
            parameters for each geometrical primitive are defined by each
            line in the 'data' parameter. For points , they are represented
            by a matrix where each row gives the point's row and column, in
            this order. For lines , they are drawn with the same convention
            used by points, with a straight line connecting them in the
            order given by the data matrix. For rectangles and filled
            rectangles , each row in the data matrix gives the two points of
            the diagonal of the rectangle, where the points use the same
            row, column convention.
        - Examples
            #
            #   example 1
            #
            f=to_uint8(zeros((3,5)))
            pcoords=uint16([[0,2,4],
                            [0,0,2]])
            pvalue=uint16([1,2,3])
            print mmdrawv(f,pcoords,pvalue,'point')
            print mmdrawv(f,pcoords,pvalue,'line')
            rectcoords=uint16([[0],
                               [0],
                               [3],
                               [2]])
            print mmdrawv(f,rectcoords, uint16(5), 'rect')
            #
            #   example 2
            #
            f=mmreadgray('blob3.tif')
            pc=mmblob(mmlabel(f),'centroid','data')
            lines=mmdrawv(mmintersec(f,0),transpose(pc),to_uint8(1),'line')
            mmshow(f,lines)
    """
    from numpy import array, newaxis, zeros, Int, put, ravel, arange, floor
    from string import upper

    GEOM  = upper(GEOM)
    data  = array(data)
    value = array(value)
    y     = array(f)
    lin, col = data[1,:], data[0,:]
    i = lin*f.shape[1] + col; i = i.astype(Int)
    if len(f.shape) == 1: f = f[newaxis,:]
    if value.shape == (): value = value + zeros(lin.shape)
    if len(lin) != len(value):
        print 'Number of points must match n. of colors.'
        return None
    if GEOM == 'POINT':
        put(ravel(y), i, value)
    elif GEOM == 'LINE':
        for k in range(len(value)-1):
            delta = 1.*(lin[k+1]-lin[k])/(1e-10 + col[k+1]-col[k])
            if abs(delta) <= 1:
                if col[k] < col[k+1]: x_ = arange(col[k],col[k+1]+1)
                else                : x_ = arange(col[k+1],col[k]+1)
                y_ = floor(delta*(x_-col[k]) + lin[k] + 0.5)
            else:
                if lin[k] < lin[k+1]: y_ = arange(lin[k],lin[k+1]+1)
                else                : y_ = arange(lin[k+1],lin[k]+1)
                x_ = floor((y_-lin[k])/delta + col[k] + 0.5)
            i_ = y_*f.shape[1] + x_; i_ = i_.astype(Int)
            put(ravel(y), i_, value[k])
    elif GEOM == 'RECT':
        for k in range(data.shape[1]):
            d = data[:,k]
            x0,y0,x1,y1 = d[1],d[0],d[3],d[2]
            y[x0:x1,y0]   = value[k]
            y[x0:x1,y1]   = value[k]
            y[x0,y0:y1]   = value[k]
            y[x1,y0:y1+1] = value[k]
    elif GEOM == 'FRECT':
        for k in range(data.shape[1]):
            d = data[:,k]
            x0,y0,x1,y1 = d[1],d[0],d[3],d[2]
            y[x0:x1+1,y0:y1+1] = value[k]
    else:
        print "GEOM should be 'POINT', 'LINE', 'RECT', or 'FRECT'."
    return y
#
# =====================================================================
#
#   mmdtshow
#
# =====================================================================
def mmdtshow(f, n=10):
    """
        - Purpose
            Display a distance transform image with an iso-line color table.
        - Synopsis
            y = mmdtshow(f, n=10)
        - Input
            f: Gray-scale (uint8 or uint16) image. Distance transform.
            n: Boolean Default: 10. Number of iso-contours.
        - Output
            y: Gray-scale (uint8 or uint16) or binary image. Optionally
               return RGB uint8 image
        - Description
            Displays the distance transform image f (uint8 or uint16) with a
            special gray-scale color table with n pseudo-color equaly
            spaced. The final appearance of this display is similar to an
            iso-contour image display. The infinity value, which is the
            maximum level allowed in the image, is displayed as black. The
            image is displayed in the MATLAB figure only if no output
            parameter is given.
        - Examples
            #
            f=mmreadgray('blob.tif')
            fd=mmdist(f)
            mmshow(fd)
            mmdtshow(fd)
    """
    import adpil

    if (mmisbinary(f)) or (len(f.shape) != 2):
      print 'Error, mmdtshow: works only for grayscale labeled image'
      return
    y=mmgdtshow(f, n)
    adpil.adshow(y)
    return
    return y
#
# =====================================================================
#
#   mmendpoints
#
# =====================================================================
def mmendpoints(OPTION="LOOP"):
    """
        - Purpose
            Interval to detect end-points.
        - Synopsis
            Iab = mmendpoints(OPTION="LOOP")
        - Input
            OPTION: String Default: "LOOP". 'LOOP' or 'HOMOTOPIC'
        - Output
            Iab: Interval
        - Description
            mmendpoints creates an interval that is useful to detect
            end-points of curves (i.e., one pixel thick connected
            components) in binary images. It can be used to prune skeletons
            and to mark objects transforming them in a single pixel or
            closed loops if they have holes. There are two options
            available: LOOP, deletes all points but preserves loops if used
            in mmthin ; HOMOTOPIC, deletes all points but preserves the last
            single point or loops.
        - Examples
            #
            #   example 1
            #
            print mmintershow(mmendpoints())
            #
            #   example 2
            #
            print mmintershow(mmendpoints('HOMOTOPIC'))
            #
            #   example 3
            #
            f = mmreadgray('pcbholes.tif')
            mmshow(f)
            f1 = mmthin(f)
            mmshow(f1)
            f2 = mmthin(f1,mmendpoints(),20)
            mmshow(f2)
            #
            #   example 4
            #
            fn = mmthin(f1,mmendpoints('HOMOTOPIC'))
            mmshow(mmdil(fn))
    """
    from string import upper

    Iab = None
    OPTION = upper(OPTION)
    if OPTION == 'LOOP':
        Iab = mmse2hmt(mmbinary([[0,0,0],
                                  [0,1,0],
                                  [0,0,0]]),
                        mmbinary([[0,0,0],
                                  [1,0,1],
                                  [1,1,1]]))
    elif OPTION == 'HOMOTOPIC':
        Iab = mmse2hmt(mmbinary([[0,1,0],
                                  [0,1,0],
                                  [0,0,0]]),
                        mmbinary([[0,0,0],
                                  [1,0,1],
                                  [1,1,1]]))
    return Iab
#
# =====================================================================
#
#   mmero
#
# =====================================================================
def mmero(f, b=None):
    """
        - Purpose
            Erode an image by a structuring element.
        - Synopsis
            y = mmero(f, b=None)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            b: Structuring Element Default: None (3x3 elementary cross).
        - Output
            y: Image
        - Description
            mmero performs the erosion of the image f by the structuring
            element b . Erosion is a neighbourhood operator that compairs
            locally b with f , according to an inclusion rule. Since erosion
            is a fundamental operator to the construction of all other
            morphological operators, it is also called an elementary
            operator of Mathematical Morphology. When f is a gray-scale
            image , b may be a flat or non-flat structuring element.
        - Examples
            #
            #   example 1
            #
            f=mmbinary([
               [1, 1, 1, 0, 0, 1, 1],
               [1, 0, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 0, 0]])
            b=mmbinary([1, 1, 0])
            mmero(f,b)
            f=to_uint8([
               [ 0,   1,  2, 50,  4,  5],
               [ 2,   3,  4,  0,  0,  0],
               [12, 255, 14, 15, 16, 17]])
            mmero(f,b)
            #
            #   example 2
            #
            f=mmbinary(mmreadgray('blob.tif'))
            bimg=mmbinary(mmreadgray('blob1.tif'))
            b=mmimg2se(bimg)
            g=mmero(f,b)
            mmshow(f)
            mmshow(g)
            mmshow(g,mmgradm(f))
            #
            #   example 3
            #
            f=mmreadgray('pcb_gray.tif')
            b=mmsedisk(3)
            mmshow(f)
            mmshow(mmero(f,b))
    """

    if b is None: b = mmsecross()
    y = mmneg(mmdil(mmneg(f),mmsereflect(b)))
    return y
#
# =====================================================================
#
#   mmfreedom
#
# =====================================================================
def mmfreedom(L=5):
    """
        - Purpose
            Control automatic data type conversion.
        - Synopsis
            Y = mmfreedom(L=5)
        - Input
            L: Double Default: 5. level of FREEDOM: 0, 1 or 2. If the input
               parameter is omitted, the current level is returned.
        - Output
            Y: Double current FREEDOM level
        - Description
            mmfreedom controls the automatic data type conversion. There are
            3 possible levels, called FREEDOM levels, for automatic
            conversion: 0 - image type conversion is not allowed; 1- image
            type conversion is allowed, but a warning is sent for each
            conversion; 2- image type conversion is allowed without warning.
            The FREEDOM levels are set or inquired by mmfreedom . If an
            image is not in the required datatype, than it should be
            converted to the maximum and nearest pymorph Morphology Toolbox
            datatype. For example, if an image is in int32 and a
            morphological gray-scale processing that accepts only binary,
            uint8 or uint16 images, is required, it will be converted to
            uint16. Another example, if a binary image should be added to a
            uint8 image, the binary image will be converted to uint8. In
            cases of operators that have as parameters an image and a
            constant, the type of the image should be kept as reference,
            while the type of the constant should be converted, if
            necessary.
        - Examples
            #
            #   example 1
            #
            a=mmsubm([4., 2., 1.],to_uint8([3, 2, 0]))
            print a
            print mmdatatype(a)
            #
            #   example 2
            #
            a=mmsubm([4., 2., 1], mmbinary([3, 2, 0]))
            print a
            print mmdatatype(a)
            #
            #   example 3
            #
            a=mmsubm(to_uint8([4, 3, 2, 1]), 1)
            print a
            print mmdatatype(a)
    """

    Y = -1
    return Y
#
# =====================================================================
#
#   mmgdist
#
# =====================================================================
def mmgdist(f, g, Bc=None, METRIC=None):
    """
        - Purpose
            Geodesic Distance Transform.
        - Synopsis
            y = mmgdist(f, g, Bc=None, METRIC=None)
        - Input
            f:      Binary image.
            g:      Binary image. Marker image
            Bc:     Structuring Element Default: None (3x3 elementary
                    cross). (metric for distance).
            METRIC: String Default: None. 'EUCLIDEAN' if specified.
        - Output
            y: uint16 (distance image).
        - Description
            mmgdist creates the geodesic distance image y of the binary
            image f relative to the binary image g . The value of y at the
            pixel x is the length of the smallest path between x and f . The
            distances available are based on the Euclidean metrics and on
            metrics generated by a neighbourhood graph, that is
            characterized by a connectivity rule defined by the structuring
            element Bc . The connectivity for defining the paths is
            consistent with the metrics adopted to measure their length. In
            the case of the Euclidean distance, the space is considered
            continuos and, in the other cases, the connectivity is the one
            defined by Bc .
        - Examples
            #
            #   example 1
            #
            f=mmbinary([
             [1,1,1,1,1,1],
             [1,1,1,0,0,1],
             [1,0,1,0,0,1],
             [1,0,1,1,0,0],
             [0,0,1,1,1,1],
             [0,0,0,1,1,1]])
            g=mmbinary([
             [0,0,0,0,0,0],
             [1,1,0,0,0,0],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0],
             [0,0,0,0,0,1]])
            y=mmgdist(f,g,mmsecross())
            print y
            #
            #   example 2
            #
            f=mmreadgray('maze_bw.tif')
            g=mmintersec(f,0)
            g=mmdrawv(g,uint16([[2],[2],[6],[6]]),uint16(1),'frect')
            y=mmgdist(f,g,mmsebox(),'EUCLIDEAN')
            mmshow(f,g)
            mmdtshow(y,200)
    """

    if Bc is None: Bc = mmsecross()
    assert METRIC is None,'Does not support EUCLIDEAN'
    fneg,gneg = mmneg(f),mmneg(g)
    y = mmgray(gneg,'uint16',1)
    ero = mmintersec(y,0)
    aux = y
    i = 1
    while not mmisequal(ero,aux):
        aux = ero
        ero = mmcero(gneg,fneg,Bc,i)
        y = mmaddm(y,mmgray(ero,'uint16',1))
        i = i + 1
    y = mmunion(y,mmgray(ero,'uint16'))
    return y
#
# =====================================================================
#
#   mmgradm
#
# =====================================================================
def mmgradm(f, Bdil=None, Bero=None):
    """
        - Purpose
            Morphological gradient.
        - Synopsis
            y = mmgradm(f, Bdil=None, Bero=None)
        - Input
            f:    Gray-scale (uint8 or uint16) or binary image.
            Bdil: Structuring Element Default: None (3x3 elementary cross).
                  for the dilation.
            Bero: Structuring Element Default: None (3x3 elementary cross).
                  for the erosion.
        - Output
            y: Gray-scale (uint8 or uint16) or binary image. (same type of f
               ).
        - Description
            mmgradm creates the image y by the subtraction of the erosion of
            the image f by Bero of the dilation of f by Bdil .
        - Examples
            #
            #   example 1
            #
            a = mmreadgray('small_bw.tif')
            b = mmgradm(a)
            mmshow(a)
            mmshow(b)
            #
            #   example 2
            #
            c=mmgradm(a,mmsecross(0),mmsecross())
            d=mmgradm(a,mmsecross(),mmsecross(0))
            mmshow(a,c)
            mmshow(a,d)
            #
            #   example 3
            #
            a = mmreadgray('bloodcells.tif')
            b = mmgradm(a)
            mmshow(a)
            mmshow(b)
    """

    if Bdil is None: Bdil = mmsecross()
    if Bero is None: Bero = mmsecross()
    y = mmsubm(mmdil(f,Bdil),mmero(f,Bero))
    return y
#
# =====================================================================
#
#   mmgrain
#
# =====================================================================
def mmgrain(fr, f, measurement, option="image"):
    """
        - Purpose
            Gray-scale statistics for each labeled region.
        - Synopsis
            y = mmgrain(fr, f, measurement, option="image")
        - Input
            fr:          Gray-scale (uint8 or uint16) image. Labeled image,
                         to define the regions. Label 0 is the background
                         region.
            f:           Gray-scale (uint8 or uint16) image. To extract the
                         measuremens.
            measurement: String Default: "". Choose the measure to compute:
                         'max', 'min', 'median', 'mean', 'sum', 'std',
                         'std1'.
            option:      String Default: "image". Output format: 'image':
                         results as a gray-scale mosaic image (uint16);
                         'data': results a column vector of measurements
                         (double).
        - Output
            y: Gray-scale (uint8 or uint16) image. Or a column vector
               (double) with gray-scale statistics per region.
        - Description
            Computes gray-scale statistics of each grain in the image. The
            grains regions are specified by the labeled image fr and the
            gray-scale information is specified by the image f . The
            statistics to compute is specified by the parameter measurement
            , which has the same options as in function mmstats . The
            parameter option defines: ('image') if the output is an uint16
            image where each label value is changed to the measurement
            value, or ('data') a double column vector. In this case, the
            first element (index 1) is the measurement of region 1. The
            region with label zero is not measure as it is normally the
            background.
        - Examples
            #
            #   example 1
            #
            f=to_uint8([range(6),range(6),range(6)])
            fr=mmlabelflat(f)
            mmgrain(fr,f,'sum','data')
            mmgrain(fr,f,'sum')
            #
            #   example 2
            #
            f=mmreadgray('astablet.tif')
            g=mmgradm(f)
            marker=mmregmin(mmclose(g))
            ws=mmcwatershed(g,marker,mmsebox(),'regions')
            g=mmgrain(ws,f,'mean')
            mmshow(f)
            mmshow(g)
    """
    from numpy import newaxis, ravel, zeros, sum, nonzero, put, take, array
    from MLab import mean, std
    from string import upper

    measurement = upper(measurement)
    option      = upper(option)
    if len(fr.shape) == 1: fr = fr[newaxis,:]
    n = max(ravel(fr))
    if option == 'DATA': y = []
    else               : y = zeros(fr.shape)
    if measurement == 'MAX':
        for i in range(1,n+1):
            aux = fr==i
            val = max(ravel(aux*f))
            if option == 'DATA': y.append(val)
            else               : put(ravel(y), nonzero(ravel(aux)), val)
    elif measurement == 'MIN':
        for i in range(1,n+1):
            aux = fr==i
            lin = ravel(aux*f)
            ind = nonzero(ravel(aux))
            val = min(take(lin,ind))
            if option == 'DATA': y.append(val)
            else               : put(ravel(y), ind, val)
    elif measurement == 'SUM':
        for i in range(1,n+1):
            aux = fr==i
            val = sum(ravel(aux*f))
            if option == 'DATA': y.append(val)
            else               : put(ravel(y), nonzero(ravel(aux)), val)
    elif measurement == 'MEAN':
        for i in range(1,n+1):
            aux = fr==i
            ind = nonzero(ravel(aux))
            val = mean(take(ravel(aux*f), ind))
            if option == 'DATA': y.append(val)
            else               : put(ravel(y), ind, val)
    elif measurement == 'STD':
        for i in range(1,n+1):
            aux = fr==i
            ind = nonzero(ravel(aux))
            v   = take(ravel(aux*f), ind)
            if len(v) < 2: val = 0
            else         : val = std(v)
            if option == 'DATA': y.append(val)
            else               : put(ravel(y), ind, val)
    elif measurement == 'STD1':
        print "'STD1' is not implemented"
    else:
        print "Measurement should be 'MAX', 'MIN', 'MEAN', 'SUM', 'STD', 'STD1'."
    if option == 'DATA':
        y = array(y)
        if len(y.shape) == 1: y = y[:,newaxis]
    return y
#
# =====================================================================
#
#   mmgray
#
# =====================================================================
def mmgray(f, TYPE="uint8", k1=None):
    """
        - Purpose
            Convert a binary image into a gray-scale image.
        - Synopsis
            y = mmgray(f, TYPE="uint8", k1=None)
        - Input
            f:    Binary image.
            TYPE: String Default: "uint8". 'uint8', 'uint16', or 'int32'.
            k1:   Non-negative integer. Default: None (Maximum pixel level
                  in pixel type).
        - Output
            y: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image.
        - Description
            mmgray converts a binary image into a gray-scale image of a
            specified data type. The value k1 is assigned to the 1 pixels of
            f , while the 0 pixels are assigned to the minimum value
            associated to the specified data type.
        - Examples
            #
            b=mmbinary([0, 1, 0, 1])
            print b
            c=mmgray(b)
            print c
            d=mmgray(b,'uint8',100)
            print d
            e=mmgray(b,'uint16')
            print e
            f=mmgray(b,'int32',0)
            print f
    """
    from numpy import array
    if k1 is None: k1 = mmmaxleveltype(TYPE)
    if type(f) is list: f = mmbinary(f)
    assert mmis(f,'binary'), 'f must be binary'
    if k1==None:
        k1=mmmaxleveltype(TYPE)
    if   TYPE == 'uint8' : y = to_uint8(f*k1)
    elif TYPE == 'uint16': y = uint16(f*k1)
    elif TYPE == 'int32' : y = int32(f*k1) - int32(mmneg(f)*mmmaxleveltype(TYPE))
    else:
        assert 0, 'type not supported:'+TYPE
    return y
#
# =====================================================================
#
#   mmhmin
#
# =====================================================================
def mmhmin(f, h=1, Bc=None):
    """
        - Purpose
            Remove basins with contrast less than h.
        - Synopsis
            y = mmhmin(f, h=1, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) image.
            h:  Default: 1. Contrast parameter.
            Bc: Structuring Element Default: None (3x3 elementary cross).
                Structuring element (connectivity).
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmhmin sup-reconstructs the gray-scale image f from the marker
            created by the addition of the positive integer value h to f ,
            using the connectivity Bc . This operator removes connected
            basins with contrast less than h . This function is very userful
            for simplifying the basins of the image.
        - Examples
            #
            #   example 1
            #
            a = to_uint8([
                [10,   3,   6,  18,  16,  15,  10],
                [10,   9,   6,  18,   6,   5,  10],
                [10,   9,   9,  15,   4,   9,  10],
                [10,  10,  10,  10,  10,  10,  10]])
            print mmhmin(a,1,mmsebox())
            #
            #   example 2
            #
            f = mmreadgray('r4x2_256.tif')
            mmshow(f)
            fb = mmhmin(f,70)
            mmshow(fb)
            mmshow(mmregmin(fb))
    """

    if Bc is None: Bc = mmsecross()
    g = mmaddm(f,h)
    y = mmsuprec(g,f,Bc);
    return y
#
# =====================================================================
#
#   mmvdome
#
# =====================================================================
def mmvdome(f, v=1, Bc=None):
    """
        - Purpose
            Obsolete, use mmvmax.
        - Synopsis
            y = mmvdome(f, v=1, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) image.
            v:  Default: 1. Volume parameter.
            Bc: Structuring Element Default: None (3x3 elementary cross).
                Structuring element (connectivity).
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            The correct name for this operator mmvdome is mmvmax.

    """

    if Bc is None: Bc = mmsecross()
    y = mmhmax(f,v,Bc);
    return y
#
# =====================================================================
#
#   mmvmax
#
# =====================================================================
def mmvmax(f, v=1, Bc=None):
    """
        - Purpose
            Remove domes with volume less than v.
        - Synopsis
            y = mmvmax(f, v=1, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) image.
            v:  Default: 1. Volume parameter.
            Bc: Structuring Element Default: None (3x3 elementary cross).
                Structuring element (connectivity).
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmvmax This operator removes connected domes with volume less
            than v . This function is very similar to mmhmax , but instead
            of using a gray scale criterion (contrast) for the dome, it uses
            a volume criterion.
        - Examples
            #
            #   example 1
            #
            a = to_uint8([
                [4,  3,  6,  1,  3,  5,  2],
                [2,  9,  6,  1,  6,  7,  3],
                [8,  9,  3,  2,  4,  9,  4],
                [3,  1,  2,  1,  2,  4,  2]])
            print mmvmax(a,10,mmsebox())
            #
            #   example 2
            #
            f = mmreadgray('astablet.tif')
            mmshow(f)
            fb = mmvmax(f,80000)
            mmshow(fb)
            mmshow(mmregmax(fb))
    """

    if Bc is None: Bc = mmsecross()
    print 'Not implemented yet'
    return None
    return y
#
# =====================================================================
#
#   mmhmax
#
# =====================================================================
def mmhmax(f, h=1, Bc=None):
    """
        - Purpose
            Remove peaks with contrast less than h.
        - Synopsis
            y = mmhmax(f, h=1, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) image.
            h:  Default: 1. Contrast parameter.
            Bc: Structuring Element Default: None (3x3 elementary cross).
                Structuring element ( connectivity).
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmhmax inf-reconstructs the gray-scale image f from the marker
            created by the subtraction of the positive integer value h from
            f , using connectivity Bc . This operator removes connected
            peaks with contrast less than h .
        - Examples
            #
            #   example 1
            #
            a = to_uint8([
                [4,   3,   6,  1,  3,  5,  2],
                [2,   9,   6,  1,  6,  7,  3],
                [8,   9,   3,  2,  4,  9,  4],
                [3,   1,   2,  1,  2,  4,  2]])
            print mmhmax(a,2,mmsebox())
            #
            #   example 2
            #
            f = mmreadgray('r4x2_256.tif')
            mmshow(f)
            fb = mmhmax(f,50)
            mmshow(fb)
            mmshow(mmregmax(fb))
    """

    if Bc is None: Bc = mmsecross()
    g = mmsubm(f,h)
    y = mminfrec(g,f,Bc);
    return y
#
# =====================================================================
#
#   mmhomothick
#
# =====================================================================
def mmhomothick():
    """
        - Purpose
            Interval for homotopic thickening.
        - Synopsis
            Iab = mmhomothick()

        - Output
            Iab: Interval
        - Description
            mmhomothick creates an interval that is useful for the homotopic
            (i.e., that conserves the relation between objects and holes)
            thickening of binary images.
        - Examples
            #
            print mmintershow(mmhomothick())
    """

    Iab = mmse2hmt(mmbinary([[1,1,1],
                             [0,0,0],
                             [0,0,0]]),
                   mmbinary([[0,0,0],
                             [0,1,0],
                             [1,1,1]]))
    return Iab
#
# =====================================================================
#
#   mmhomothin
#
# =====================================================================
def mmhomothin():
    """
        - Purpose
            Interval for homotopic thinning.
        - Synopsis
            Iab = mmhomothin()

        - Output
            Iab: Interval
        - Description
            mmhomothin creates an interval that is useful for the homotopic
            (i.e., that conserves the relation between objects and holes)
            thinning of binary images.

    """

    Iab = mmse2hmt(mmbinary([[0,0,0],
                             [0,1,0],
                             [1,1,1]]),
                   mmbinary([[1,1,1],
                             [0,0,0],
                             [0,0,0]]))
    return Iab
#
# =====================================================================
#
#   mmimg2se
#
# =====================================================================
def mmimg2se(fd, FLAT="FLAT", f=None):
    """
        - Purpose
            Create a structuring element from a pair of images.
        - Synopsis
            B = mmimg2se(fd, FLAT="FLAT", f=None)
        - Input
            fd:   Binary image. The image is in the matrix format where the
                  origin (0,0) is at the matrix center.
            FLAT: String Default: "FLAT". 'FLAT' or 'NON-FLAT'.
            f:    Unsigned gray-scale (uint8 or uint16), signed (int32) or
                  binary image. Default: None.
        - Output
            B: Structuring Element
        - Description
            mmimg2se creates a flat structuring element B from the binary
            image fd or creates a non-flat structuring element b from the
            binary image fd and the gray-scale image f . fd represents the
            domain of b and f represents the image of the points in fd .
        - Examples
            #
            #   example 1
            #
            a = mmimg2se(mmbinary([
              [0,1,0],
              [1,1,1],
              [0,1,0]]))
            print mmseshow(a)
            #
            #   example 2
            #
            b = mmbinary([
              [0,1,1,1],
              [1,1,1,0]])
            b1 = mmimg2se(b)
            print mmseshow(b1)
            #
            #   example 3
            #
            c = mmbinary([
              [0,1,0],
              [1,1,1],
              [0,1,0]])
            d = int32([
              [0,0,0],
              [0,1,0],
              [0,0,0]])
            e = mmimg2se(c,'NON-FLAT',d)
            print mmseshow(e)
    """
    from string import upper
    from numpy import choose, ones

    assert mmisbinary(fd),'First parameter must be binary'
    FLAT = upper(FLAT)
    if FLAT == 'FLAT':
        return mmseshow(fd)
    else:
        B = choose(fd, (mmlimits(int32([0]))[0]*ones(fd.shape),f) )
    B = mmseshow(int32(B),'NON-FLAT')
    return B
#
# =====================================================================
#
#   mminfcanon
#
# =====================================================================
def mminfcanon(f, Iab, theta=45, DIRECTION="CLOCKWISE"):
    """
        - Purpose
            Intersection of inf-generating operators.
        - Synopsis
            y = mminfcanon(f, Iab, theta=45, DIRECTION="CLOCKWISE")
        - Input
            f:         Binary image.
            Iab:       Interval
            theta:     Double Default: 45. Degrees of rotation: 45, 90, or
                       180.
            DIRECTION: String Default: "CLOCKWISE". 'CLOCKWISE' or '
                       ANTI-CLOCKWISE'
        - Output
            y: Binary image.
        - Description
            mminfcanon creates the image y by computing intersections of
            transformations of the image f by inf-generating (i.e., dual of
            the hit-or-miss) operators. These inf-generating operators are
            characterized by rotations (in the clockwise or anti-clockwise
            direction) of theta degrees of the interval Iab .

    """
    from string import upper

    DIRECTION = upper(DIRECTION)            
    y = mmunion(f,1)
    for t in range(0,360,theta):
        Irot = mminterot( Iab, t, DIRECTION )
        y = mmintersec( y, mminfgen(f, Irot))
    return y
#
# =====================================================================
#
#   mminfgen
#
# =====================================================================
def mminfgen(f, Iab):
    """
        - Purpose
            Inf-generating.
        - Synopsis
            y = mminfgen(f, Iab)
        - Input
            f:   Binary image.
            Iab: Interval
        - Output
            y: Binary image.
        - Description
            mminfgen creates the image y by computing the transformation of
            the image f by the inf-generating operator (or dual of the
            hit-or-miss) characterized by the interval Iab .

    """

    A,Bc = Iab
    y = mmunion(mmdil( f, A),mmdil( mmneg(f), Bc))
    return y
#
# =====================================================================
#
#   mminfrec
#
# =====================================================================
def mminfrec(f, g, bc=None):
    """
        - Purpose
            Inf-reconstruction.
        - Synopsis
            y = mminfrec(f, g, bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) or binary image. Marker image.
            g:  Gray-scale (uint8 or uint16) or binary image. Conditioning
                image.
            bc: Structuring Element Default: None (3x3 elementary cross).
                Structuring element ( connectivity).
        - Output
            y: Image
        - Description
            mminfrec creates the image y by an infinite number of recursive
            iterations (iterations until stability) of the dilation of f by
            bc conditioned to g . We say the y is the inf-reconstruction of
            g from the marker f . For algorithms and applications, see
            Vinc:93b .
        - Examples
            #
            #   example 1
            #
            g=mmreadgray('text_128.tif')
            f=mmero(g,mmseline(9,90))
            y=mminfrec(f,g,mmsebox())
            mmshow(g)
            mmshow(f)
            mmshow(y)
            #
            #   example 2
            #
            g=mmneg(mmreadgray('n2538.tif'))
            f=mmintersec(g,0)
            f=mmdraw(f,'LINE:40,30,60,30:END')
            y30=mmcdil(f,g,mmsebox(),30)
            y=mminfrec(f,g,mmsebox())
            mmshow(g)
            mmshow(f)
            mmshow(y30)
            mmshow(y)
    """
    from numpy import product
    if bc is None: bc = mmsecross()
    n = product(f.shape)
    y = mmcdil(f,g,bc,n);
    return y
#
# =====================================================================
#
#   mminpos
#
# =====================================================================
def mminpos(f, g, bc=None):
    """
        - Purpose
            Minima imposition.
        - Synopsis
            y = mminpos(f, g, bc=None)
        - Input
            f:  Binary image. Marker image.
            g:  Gray-scale (uint8 or uint16) image. input image.
            bc: Structuring Element Default: None (3x3 elementary cross).
                (connectivity).
        - Output
            y: Gray-scale (uint8 or uint16) image.
        - Description
            Minima imposition on g based on the marker f . mminpos creates
            an image y by filing the valleys of g that does not cover the
            connect components of f . A remarkable property of y is that its
            regional minima are exactly the connect components of g .

    """

    if bc is None: bc = mmsecross()
    assert mmisbinary(f),'First parameter must be binary image'
    fg = mmgray(mmneg(f),mmdatatype(g))
    k1 = mmlimits(g)[1] - 1
    y = mmsuprec(fg, mmintersec(mmunion(g, 1), k1, fg), bc)
    return y
#
# =====================================================================
#
#   mminterot
#
# =====================================================================
def mminterot(Iab, theta=45, DIRECTION="CLOCKWISE"):
    """
        - Purpose
            Rotate an interval
        - Synopsis
            Irot = mminterot(Iab, theta=45, DIRECTION="CLOCKWISE")
        - Input
            Iab:       Interval
            theta:     Double Default: 45. Degrees of rotation. Available
                       values are multiple of 45 degrees.
            DIRECTION: String Default: "CLOCKWISE". 'CLOCKWISE' or '
                       ANTI-CLOCKWISE'.
        - Output
            Irot: Interval
        - Description
            mminterot rotates the interval Iab by an angle theta .
        - Examples
            #
            b1 = mmendpoints()
            b2 = mminterot(b1)
            print mmintershow(b1)
            print mmintershow(b2)
    """
    from string import upper

    DIRECTION = upper(DIRECTION)
    A,Bc = Iab
    if DIRECTION != 'CLOCKWISE':
        theta = 360 - theta
    Irot = mmse2hmt(mmserot(A, theta),
                    mmserot(Bc,theta))
    return Irot
#
# =====================================================================
#
#   mmintersec
#
# =====================================================================
def mmintersec(f1, f2, f3=None, f4=None, f5=None):
    """
        - Purpose
            Intersection of images.
        - Synopsis
            y = mmintersec(f1, f2, f3=None, f4=None, f5=None)
        - Input
            f1: Gray-scale (uint8 or uint16) or binary image.
            f2: Gray-scale (uint8 or uint16) or binary image. Or constant.
            f3: Gray-scale (uint8 or uint16) or binary image. Default: None.
                Or constant.
            f4: Gray-scale (uint8 or uint16) or binary image. Default: None.
                Or constant.
            f5: Gray-scale (uint8 or uint16) or binary image. Default: None.
                Or constant.
        - Output
            y: Image
        - Description
            mmintersec creates the image y by taking the pixelwise minimum
            between the images f1, f2, f3, f4, and f5 . When f1, f2, f3, f4,
            and f5 are binary images, y is the intersection of them.
        - Examples
            #
            #   example 1
            #
            f=to_uint8([255,  255,    0,   10,    0,   255,   250])
            g=to_uint8([ 0,    40,   80,   140,  250,    10,    30])
            print mmintersec(f, g)
            print mmintersec(f, 0)
            #
            #   example 2
            #
            a = mmreadgray('form-ok.tif')
            b = mmreadgray('form-1.tif')
            c = mmintersec(a,b)
            mmshow(a)
            mmshow(b)
            mmshow(c)
            #
            #   example 3
            #
            d = mmreadgray('tplayer1.tif')
            e = mmreadgray('tplayer2.tif')
            f = mmreadgray('tplayer3.tif')
            g = mmintersec(d,e,f)
            mmshow(d)
            mmshow(e)
            mmshow(f)
            mmshow(g)
    """
    from numpy import minimum

    y = minimum(f1,f2)
    if f3 != None: y = minimum(y,f3)
    if f4 != None: y = minimum(y,f4)
    if f5 != None: y = minimum(y,f5)
    y = y.astype(f1.dtype)
    return y
#
# =====================================================================
#
#   mmintershow
#
# =====================================================================
def mmintershow(Iab):
    """
        - Purpose
            Visualize an interval.
        - Synopsis
            s = mmintershow(Iab)
        - Input
            Iab: Interval
        - Output
            s: String ( representation of the interval).
        - Description
            mmintershow creates a representation for an interval using 0, 1
            and . ( don't care).
        - Examples
            #
            print mmintershow(mmhomothin())
    """
    from numpy import array, product, reshape, choose
    from string import join

    assert (type(Iab) is tuple) and (len(Iab) == 2),'not proper fortmat of hit-or-miss template'
    A,Bc = Iab
    S = mmseunion(A,Bc)
    Z = mmintersec(S,0)
    n = product(S.shape)
    one  = reshape(array(n*'1','c'),S.shape)
    zero = reshape(array(n*'0','c'),S.shape)
    x    = reshape(array(n*'.','c'),S.shape)
    saux = choose( S + mmseunion(Z,A), ( x, zero, one))
    s = ''
    for i in range(saux.shape[0]):
        s=s+(join(list(saux[i]))+' \n')
    return s
#
# =====================================================================
#
#   mmis
#
# =====================================================================
def mmis(f1, oper, f2=None, oper1=None, f3=None):
    """
        - Purpose
            Verify if a relationship among images is true or false.
        - Synopsis
            y = mmis(f1, oper, f2=None, oper1=None, f3=None)
        - Input
            f1:    Gray-scale (uint8 or uint16) or binary image.
            oper:  String relationship from: '==', '~=', '<','<=', '>',
                   '>=', 'binary', 'gray'.
            f2:    Gray-scale (uint8 or uint16) or binary image. Default:
                   None.
            oper1: String Default: None. relationship from: '==', '~=',
                   '<','<=', '>', '>='.
            f3:    Gray-scale (uint8 or uint16) or binary image. Default:
                   None.
        - Output
            y: Bool value: 0 or 1
        - Description
            Verify if the property or relatioship between images is true or
            false. The result is true if the relationship is true for all
            the pixels in the image, and false otherwise. (Obs: This
            function replaces mmis equal, mmis lesseq, mmis binary ).
        - Examples
            #
            fbin=mmbinary([0, 1])
            f1=to_uint8([1, 2, 3])
            f2=to_uint8([2, 2, 3])
            f3=to_uint8([2, 3, 4])
            mmis(fbin,'binary')
            mmis(f1,'gray')
            mmis(f1,'==',f2)
            mmis(f1,'<',f3)
            mmis(f1,'<=',f2)
            mmis(f1,'<=',f2,'<=',f3)
    """
    from string import upper

    if f2 == None:
        oper=upper(oper);
        if   oper == 'BINARY': return mmisbinary(f1)
        elif oper == 'GRAY'  : return not mmisbinary(f1)
        else:
            assert 0,'oper should be BINARY or GRAY, was'+oper
    elif oper == '==':    y = mmisequal(f1, f2)
    elif oper == '~=':    y = not mmisequal(f1,f2)
    elif oper == '<=':    y = mmislesseq(f1,f2)
    elif oper == '>=':    y = mmislesseq(f2,f1)
    elif oper == '>':     y = mmisequal(mmneg(mmthreshad(f2,f1)),mmbinary(1))
    elif oper == '<':     y = mmisequal(mmneg(mmthreshad(f1,f2)),mmbinary(1))
    else:
        assert 0,'oper must be one of: ==, ~=, >, >=, <, <=, it was:'+oper
    if oper1 != None:
        if   oper1 == '==': y = y and mmisequal(f2,f3)
        elif oper1 == '~=': y = y and (not mmisequal(f2,f3))
        elif oper1 == '<=': y = y and mmislesseq(f2,f3)
        elif oper1 == '>=': y = y and mmislesseq(f3,f2)
        elif oper1 == '>':  y = y and mmisequal(mmneg(mmthreshad(f3,f2)),mmbinary(1))
        elif oper1 == '<':  y = y and mmisequal(mmneg(mmthreshad(f2,f3)),mmbinary(1))
        else:
            assert 0,'oper1 must be one of: ==, ~=, >, >=, <, <=, it was:'+oper1
    return y
#
# =====================================================================
#
#   mmisbinary
#
# =====================================================================
def mmisbinary(f):
    """
        - Purpose
            Check for binary image
        - Synopsis
            bool = mmisbinary(f)
        - Input
            f:
        - Output
            bool: Boolean
        - Description
            mmisbinary returns TRUE(1) if the datatype of the input image is
            binary. A binary image has just the values 0 and 1.
        - Examples
            #
            a=to_uint8([0, 1, 0, 1])
            print mmisbinary(a)
            b=(a)
            print mmisbinary(b)
    """
    return type(f) is type(mmbinary([1])) and f.dtype == bool
#
# =====================================================================
#
#   mmisequal
#
# =====================================================================
def mmisequal(f1, f2, MSG=None):
    """
        - Purpose
            Verify if two images are equal
        - Synopsis
            bool = mmisequal(f1, f2)
        - Input
            f1:  Unsigned gray-scale (uint8 or uint16), signed (int32) or
                 binary image.
            f2:  Unsigned gray-scale (uint8 or uint16), signed (int32) or
                 binary image.
        - Output
            bool: Boolean
        - Description
            mmisequal compares the images f1 and f2 and returns true (1), if
            f1(x)=f2(x) , for all pixel x , and false (0), otherwise.
        - Examples
            #
            f1 = to_uint8(arrayrange(4))
            print f1
            f2 = to_uint8([9, 5, 3, 3])
            print f2
            f3 = f1
            mmisequal(f1,f2)
            mmisequal(f1,f3)
    """
    from numpy import ravel, alltrue, array

    bool = alltrue(ravel(f1==f2))
    bool1 = 1
    if type(f1) is type(array([1])):
        bool1 = type(f1) is type(f2)
        bool1 = bool1 and ((f1.dtype == f2.dtype))
    if MSG != None:
        if bool:
            if bool1:
                print 'OK: ', MSG
            else:
                print 'WARNING:', MSG
        else:
            print 'ERROR: ', MSG
    return bool
#
# =====================================================================
#
#   mmislesseq
#
# =====================================================================
def mmislesseq(f1, f2, MSG=None):
    """
        - Purpose
            Verify if one image is less or equal another (is beneath)
        - Synopsis
            bool = mmislesseq(f1, f2)
        - Input
            f1:  Gray-scale (uint8 or uint16) or binary image.
            f2:  Gray-scale (uint8 or uint16) or binary image.
        - Output
            bool: Boolean
        - Description
            mmislesseq compares the images f1 and f2 and returns true (1),
            if f1(x) <= f2(x) , for every pixel x, and false (0), otherwise.
        - Examples
            #
            f1 = to_uint8([0, 1, 2, 3])
            f2 = to_uint8([9, 5, 3, 3])
            print mmislesseq(f1,f2)
            print mmislesseq(f2,f1)
            print mmislesseq(f1,f1)
    """
    from numpy import ravel

    bool = min(ravel(f1<=f2))
    return bool
#
# =====================================================================
#
#   mmlabelflat
#
# =====================================================================
def mmlabelflat(f, Bc=None, _lambda=0):
    """
        - Purpose
            Label the flat zones of gray-scale images.
        - Synopsis
            y = mmlabelflat(f, Bc=None, _lambda=0)
        - Input
            f:       Gray-scale (uint8 or uint16) or binary image.
            Bc:      Structuring Element Default: None (3x3 elementary
                     cross). ( connectivity).
            _lambda: Default: 0. Connectivity given by |f(q)-f(p)|<=_lambda.
        - Output
            y: Image If number of labels is less than 65535, the data type
               is uint16, otherwise it is int32.
        - Description
            mmlabelflat creates the image y by labeling the flat zones of f
            , according to the connectivity defined by the structuring
            element Bc . A flat zone is a connected region of the image
            domain in which all the pixels have the same gray-level
            (lambda=0 ). When lambda is different than zero, a quasi-flat
            zone is detected where two neighboring pixels belong to the same
            region if their difference gray-levels is smaller or equal
            lambda . The minimum label of the output image is 1 and the
            maximum is the number of flat-zones in the image.
        - Examples
            #
            #   example 1
            #
            f=to_uint8([
               [5,5,8,3,0],
               [5,8,8,0,2]])
            g=mmlabelflat(f)
            print g
            g1=mmlabelflat(f,mmsecross(),2)
            print g1
            #
            #   example 2
            #
            f=mmreadgray('blob.tif')
            d=mmdist(f,mmsebox(),'euclidean')
            g= d /8
            mmshow(g)
            fz=mmlabelflat(g,mmsebox());
            mmlblshow(fz)
            print mmstats(fz,'max')
            #
            #   example 3
            #
            f=mmreadgray('pcb_gray.tif')
            g=mmlabelflat(f,mmsebox(),3)
            mmshow(f)
            mmlblshow(g)
    """
    from numpy import allclose, ravel, nonzero, array
    if Bc is None: Bc = mmsecross()
    zero = mmbinary(mmsubm(f,f))       # zero image
    faux = mmneg(zero)
    r = array(zero)
    label = 1
    y = mmgray( zero,'uint16',0)          # zero image (output)
    while not allclose(faux,0):
        x=nonzero(ravel(faux))[0]        # get first unlabeled pixel
        fmark = array(zero)
        fmark.flat[x] = 1                # get the first unlabeled pixel
        f2aux = mmcmp( f, '==', ravel(f)[x])
        r = mminfrec( fmark, f2aux, Bc)  # detects all pixels connected to it
        faux = mmsubm( faux, r)          # remove them from faux
        r = mmgray( r,'uint16',label)    # label them with the value label
        y = mmunion( y, r)               # merge them with the labeled image
        label = label + 1
    return y
#
# =====================================================================
#
#   mmlastero
#
# =====================================================================
def mmlastero(f, B=None):
    """
        - Purpose
            Last erosion.
        - Synopsis
            y = mmlastero(f, B=None)
        - Input
            f: Binary image.
            B: Structuring Element Default: None (3x3 elementary cross).
        - Output
            y: Binary image.
        - Description
            mmlastero creates the image y by computing the last erosion by
            the structuring element B of the image f . The objects found in
            y are the objects of the erosion by nB that can not be
            reconstructed from the erosion by (n+1)B , where n is a generic
            non negative integer. The image y is a proper subset of the
            morphological skeleton by B of f .

    """

    if B is None: B = mmsecross()
    assert mmisbinary(f),'Can only process binary images'
    dt = mmdist(f,B)
    y = mmregmax(dt,B)
    return y
#
# =====================================================================
#
#   mmlblshow
#
# =====================================================================
def mmlblshow(f, option='noborder'):
    """
        - Purpose
            Display a labeled image assigning a random color for each label.
        - Synopsis
            y = mmlblshow(f, option='noborder')
        - Input
            f:      Gray-scale (uint8 or uint16) image. Labeled image.
            option: String Default: 'noborder'. BORDER or NOBORDER: includes
                    or not a white border around each labeled region
        - Output
            y: Gray-scale (uint8 or uint16) or binary image. Optionally
               return RGB uint8 image
        - Description
            Displays the labeled image f (uint8 or uint16) with a pseudo
            color where each label appears with a random color. The image is
            displayed in the MATLAB figure only if no output parameter is
            given.
        - Examples
            #
            f=mmreadgray('blob3.tif')
            f1=mmlabel(f,mmsebox())
            mmshow(f1)
            mmlblshow(f1)
            mmlblshow(f1,'border')
    """
    import string
    import adpil

    if (mmisbinary(f)) or (len(f.shape) != 2):
      print 'Error, mmlblshow: works only for grayscale labeled image'
      return
    option = string.upper(option);
    if option == 'NOBORDER':
      border = 0.0;
    elif option == 'BORDER':
      border = 1.0;
    else:
      print 'Error: option must be BORDER or NOBORDER'
    y=mmglblshow(f, border);
    adpil.adshow(y)
    return
    return y
#
# =====================================================================
#
#   mmopen
#
# =====================================================================
def mmopen(f, b=None):
    """
        - Purpose
            Morphological opening.
        - Synopsis
            y = mmopen(f, b=None)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            b: Structuring Element Default: None (3x3 elementary cross).
        - Output
            y: Image
        - Description
            mmopen creates the image y by the morphological opening of the
            image f by the structuring element b . In the binary case, the
            opening by the structuring element B may be interpreted as the
            union of translations of B included in f . In the gray-scale
            case, there is a similar interpretation taking the functions
            umbra.
        - Examples
            #
            #   example 1
            #
            f=mmbinary(mmreadgray('blob.tif'))
            bimg=mmbinary(mmreadgray('blob1.tif'))
            b=mmimg2se(bimg)
            mmshow(f)
            mmshow(mmopen(f,b))
            mmshow(mmopen(f,b),mmgradm(f))
            #
            #   example 2
            #
            a=mmbinary(mmreadgray('pcb1bin.tif'))
            b=mmopen(a,mmsebox(2))
            c=mmopen(a,mmsebox(4))
            mmshow(a)
            mmshow(b)
            mmshow(c)
            #
            #   example 3
            #
            a=mmreadgray('astablet.tif')
            b=mmopen(a,mmsedisk(18))
            mmshow(a)
            mmshow(b)
    """

    if b is None: b = mmsecross()
    y = mmdil(mmero(f,b),b)
    return y
#
# =====================================================================
#
#   mmopenrec
#
# =====================================================================
def mmopenrec(f, bero=None, bc=None):
    """
        - Purpose
            Opening by reconstruction.
        - Synopsis
            y = mmopenrec(f, bero=None, bc=None)
        - Input
            f:    Gray-scale (uint8 or uint16) or binary image.
            bero: Structuring Element Default: None (3x3 elementary cross).
                  (erosion).
            bc:   Structuring Element Default: None (3x3 elementary cross).
                  (connectivity).
        - Output
            y: Image (same type of f ).
        - Description
            mmopenrec creates the image y by an inf-reconstruction of the
            image f from its erosion by bero , using the connectivity
            defined by Bc .

    """

    if bero is None: bero = mmsecross()
    if bc is None: bc = mmsecross()
    y = mminfrec(mmero(f,bero),f,bc)
    return y
#
# =====================================================================
#
#   mmopenrecth
#
# =====================================================================
def mmopenrecth(f, bero=None, bc=None):
    """
        - Purpose
            Open-by-Reconstruction Top-Hat.
        - Synopsis
            y = mmopenrecth(f, bero=None, bc=None)
        - Input
            f:    Gray-scale (uint8 or uint16) or binary image.
            bero: Structuring Element Default: None (3x3 elementary cross).
                  (erosion)
            bc:   Structuring Element Default: None (3x3 elementary cross).
                  ( connectivity)
        - Output
            y: Gray-scale (uint8 or uint16) or binary image. (same type of f
               ).
        - Description
            mmopenrecth creates the image y by subtracting the open by
            reconstruction of f , defined by the structuring elements bero e
            bc , of f itself.

    """

    if bero is None: bero = mmsecross()
    if bc is None: bc = mmsecross()
    y = mmsubm(f, mmopenrec( f, bero, bc))
    return y
#
# =====================================================================
#
#   mmopenth
#
# =====================================================================
def mmopenth(f, b=None):
    """
        - Purpose
            Opening Top Hat.
        - Synopsis
            y = mmopenth(f, b=None)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            b: Structuring Element Default: None (3x3 elementary cross).
               structuring element
        - Output
            y: Gray-scale (uint8 or uint16) or binary image. (same type of f
               ).
        - Description
            mmopenth creates the image y by subtracting the morphological
            opening of f by the structuring element b of f itself.
        - Examples
            #
            a = mmreadgray('keyb.tif')
            mmshow(a)
            b = mmopenth(a,mmsebox(3))
            mmshow(b)
    """

    if b is None: b = mmsecross()
    y = mmsubm(f, mmopen(f,b))
    return y
#
# =====================================================================
#
#   mmopentransf
#
# =====================================================================
def mmopentransf(f, type='OCTAGON', n=65535, Bc=None, Buser=None):
    """
        - Purpose
            Open transform.
        - Synopsis
            y = mmopentransf(f, type='OCTAGON', n=65535, Bc=None,
            Buser=None)
        - Input
            f:     Binary image.
            type:  String Default: 'OCTAGON'. Disk family: 'OCTAGON',
                   'CHESSBOARD', 'CITY-BLOCK', 'LINEAR-V', 'LINEAR-H',
                   'LINEAR-45R', 'LINEAR-45L', 'USER'.
            n:     Default: 65535. Maximum disk radii.
            Bc:    Structuring Element Default: None (3x3 elementary cross).
                   Connectivity for the reconstructive opening. Used if
                   '-REC' suffix is appended in the 'type' string.
            Buser: Structuring Element Default: None (3x3 elementary cross).
                   User disk, used if 'type' is 'USER'.
        - Output
            y: Gray-scale (uint8 or uint16) image.
        - Description
            Compute the open transform of a binary image. The value of the
            pixels in the open transform gives the largest radii of the disk
            plus 1, where the open by it is not empty at that pixel. The
            disk sequence must satisfy the following: if r > s, rB is
            sB-open, i.e. rB open by sB is equal rB. Note that the Euclidean
            disk does not satisfy this property in the discrete grid. This
            function also computes the reconstructive open transform by
            adding the suffix '-REC' in the 'type' parameter.
        - Examples
            #
            #   example 1
            #
            f = mmbinary([
                          [0,0,0,0,0,0,0,0],
                          [0,0,1,1,1,1,0,0],
                          [0,0,1,1,1,1,1,0],
                          [0,1,0,1,1,1,0,0],
                          [1,1,0,0,0,0,0,0]])
            print mmopentransf( f, 'city-block')
            print mmopentransf( f, 'linear-h')
            print mmopentransf( f, 'linear-45r')
            print mmopentransf( f, 'user',10,mmsecross(),mmbinary([0,1,1]))
            print mmopentransf( f, 'city-block-rec')
            #
            #   example 2
            #
            f=mmreadgray('numbers.tif')
            mmshow(f)
            g=mmopentransf(f,'OCTAGON')
            mmshow(g)
            #
            #   example 3
            #
            b=mmsedisk(3,'2D','OCTAGON')
            g1=mmopen(f,b)
            mmshow(g1)
            g2=mmcmp(g,'>',3)
            print mmis(g1,'==',g2)
    """
    from numpy import zeros
    from string import find, upper
    if Bc is None: Bc = mmsecross()
    if Buser is None: Buser = mmsecross()
    assert mmisbinary(f),'Error: input image is not binary'
    type = upper(type)
    rec_flag = find(type,'-REC')
    if rec_flag != -1:
        type = type[:rec_flag] # remove the -rec suffix
    flag = not ((type == 'OCTAGON')  or
                (type == 'CHESSBOARD') or
                (type == 'CITY-BLOCK'))
    if not flag:
        n  = min(n,min(f.shape))
    elif  type == 'LINEAR-H':
        se = mmbinary([1, 1, 1])
        n  = min(n,f.shape[1])
    elif  type =='LINEAR-V':
        se = mmbinary([[1],[1],[1]])
        n  = min(n,f.shape[0])
    elif  type == 'LINEAR-45R':
        se = mmbinary([[0, 0, 1],[0, 1, 0],[1, 0, 0]])
        n  = min(n,min(f.shape))
    elif  type == 'LINEAR-45L':
        se = mmbinary([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        n  = min(n,min(f.shape))
    elif  type == 'USER':
        se = Buser
        n  = min(n,min(f.shape))
    else:
        print 'Error: only accepts OCTAGON, CHESSBOARD, CITY-BLOCK, LINEAR-H, LINEAR-V, LINEAR-45R, LINEAR-45L, or USER as type, or with suffix -REC.'
        return []
    k = 0
    y = uint16(zeros(f.shape))
    a = mmbinary([1])
    z = mmbinary([0])
    while not (mmisequal(a,z) or (k>=n)):
        print 'processing r=',k
        if flag:
            a = mmopen(f,mmsesum(se,k))
        else:
            a = mmopen(f,mmsedisk(k,'2D',type))
        y = mmaddm(y, mmgray(a,'uint16',1))
        k = k+1
    if rec_flag != -1:
        y = mmgrain(mmlabel(f,Bc),y,'max')
    return y
#
# =====================================================================
#
#   mmpatspec
#
# =====================================================================
def mmpatspec(f, type='OCTAGON', n=65535, Bc=None, Buser=None):
    """
        - Purpose
            Pattern spectrum (also known as granulometric size density).
        - Synopsis
            h = mmpatspec(f, type='OCTAGON', n=65535, Bc=None, Buser=None)
        - Input
            f:     Binary image.
            type:  String Default: 'OCTAGON'. Disk family: 'OCTAGON',
                   'CHESSBOARD', 'CITY-BLOCK', 'LINEAR-V', 'LINEAR-H',
                   'LINEAR-45R', 'LINEAR-45L', 'USER'.
            n:     Default: 65535. Maximum disk radii.
            Bc:    Structuring Element Default: None (3x3 elementary cross).
                   Connectivity for the reconstructive granulometry. Used if
                   '-REC' suffix is appended in the 'type' string.
            Buser: Structuring Element Default: None (3x3 elementary cross).
                   User disk, used if 'type' is 'USER'.
        - Output
            h: Gray-scale (uint8 or uint16) or binary image. a uint16
               vector.
        - Description
            Compute the Pattern Spectrum of a binary image. See Mara:89b .
            The pattern spectrum is the histogram of the open transform, not
            taking the zero values.

    """

    if Bc is None: Bc = mmsecross()
    if Buser is None: Buser = mmsecross()
    assert mmisbinary(f),'Error: input image is not binary'
    g=mmopentransf(f,type,n,Bc,Buser)
    h=mmhistogram(g)
    h=h[1:]
    return h
#
# =====================================================================
#
#   mmreadgray
#
# =====================================================================
def mmreadgray(filename):
    """
        - Purpose
            Read an image from a commercial file format and stores it as a
            gray-scale image.
        - Synopsis
            y = mmreadgray(filename)
        - Input
            filename: String Name of file to read.
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmreadgray reads the image in filename and stores it in y , an
            uint8 gray-scale image (without colormap). If the input file is
            a color RGB image, it is converted to gray-scale using the
            equation: y = 0.2989 R + 0.587 G + 0.114 B. This functions uses
            de PIL module.
        - Examples
            #
            a=mmreadgray('cookies.tif')
            mmshow(a)
    """
    import adpil
    import numpy

    y = adpil.adread(filename)
    if (len(y.shape) == 3) and (y.shape[0] == 3):
       if numpy.alltrue(numpy.alltrue(y[0,:,:] == y[1,:,:] and
                                          y[0,:,:] == y[2,:,:])):
          y = y[0,:,:]
       else:
          print 'Warning: converting true-color RGB image to gray'
          y = mmubyte(0.2989 * y[0,:,:] + 
                      0.5870 * y[1,:,:] + 
                      0.1140 * y[2,:,:])
    elif (len(y.shape) == 2):
       pass
    else:
       raise ValueError, 'Error, it is not 2D image'
    return y
#
# =====================================================================
#
#   mmregister
#
# =====================================================================
def mmregister(code=None, file_name=None):
    """
        - Purpose
            Register the SDC Morphology Toolbox.
        - Synopsis
            s = mmregister(code=None, file_name=None)
        - Input
            code:      String Default: None. Authorization code.
            file_name: String Default: None. Filename of the license file to
                       be created.
        - Output
            s: String Message of the status of the license.
        - Description
            mmregister licenses the copy of the SDC Morphology Toolbox by
            entering the license code and the toolbox license file. If
            mmregister is called without parameters, it returns the internal
            code that must be sent for registration.

    """

    s = 'License is not required'
    return s
#
# =====================================================================
#
#   mmregmax
#
# =====================================================================
def mmregmax(f, Bc=None):
    """
        - Purpose
            Regional Maximum.
        - Synopsis
            y = mmregmax(f, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) image.
            Bc: Structuring Element Default: None (3x3 elementary cross).
                (connectivity).
        - Output
            y: Binary image.
        - Description
            mmregmax creates a binary image y by computing the regional
            maxima of f , according to the connectivity defined by the
            structuring element Bc . A regional maximum is a flat zone not
            surrounded by flat zones of higher gray values.

    """

    if Bc is None: Bc = mmsecross()
    y = mmregmin(mmneg(f),Bc)
    return y
#
# =====================================================================
#
#   mmregmin
#
# =====================================================================
def mmregmin(f, Bc=None, option="binary"):
    """
        - Purpose
            Regional Minimum (with generalized dynamics).
        - Synopsis
            y = mmregmin(f, Bc=None, option="binary")
        - Input
            f:      Gray-scale (uint8 or uint16) image.
            Bc:     Structuring Element Default: None (3x3 elementary
                    cross). (connectivity).
            option: String Default: "binary". Choose one of: BINARY: output
                    a binary image; VALUE: output a grayscale image with
                    points at the regional minimum with the pixel values of
                    the input image; DYNAMICS: output a grayscale image with
                    points at the regional minimum with its dynamics;
                    AREA-DYN: int32 image with the area-dynamics;
                    VOLUME-DYN: int32 image with the volume-dynamics.
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmregmin creates a binary image f by computing the regional
            minima of f , according to the connectivity defined by the
            structuring element Bc . A regional minimum is a flat zone not
            surrounded by flat zones of lower gray values. A flat zone is a
            maximal connected component of a gray-scale image with same
            pixel values. There are three output options: binary image;
            valued image; and generalized dynamics. The dynamics of a
            regional minima is the minimum height a pixel has to climb in a
            walk to reach another regional minima with a higher dynamics.
            The area-dyn is the minimum area a catchment basin has to raise
            to reach another regional minima with higher area-dynamics. The
            volume-dyn is the minimum volume a catchment basin has to raise
            to reach another regional minima with a higher volume dynamics.
            The dynamics concept was first introduced in Grimaud:92 and it
            is the basic notion for the hierarchical or multiscale watershed
            transform.
        - Examples
            #
            #   example 1
            #
            a = to_uint8([
                [10,  10,  10,  10,  10,  10,  10],
                [10,   9,   6,  18,   6,   5,  10],
                [10,   9,   6,  18,   6,   5,  10],
                [10,   9,   9,  15,   4,   9,  10],
                [10,   9,   9,  15,  12,  10,  10],
                [10,  10,  10,  10,  10,  10,  10]])
            print mmregmin(a)
            print mmregmin(a,mmsecross(),'value')
            print mmregmin(a,mmsecross(),'dynamics')
            #
            #   example 2
            #
            f1=mmreadgray('bloodcells.tif')
            m1=mmregmin(f1,mmsebox())
            mmshow(f1,m1)
            f2=mmhmin(f1,70)
            mmshow(f2)
            m2=mmregmin(f2,mmsebox())
            mmshow(f2,m2)
            #
            #   example 3
            #
            f=mmreadgray('cameraman.tif')
            g=mmgradm(f)
            mh=mmregmin(g,mmsecross(),'dynamics')
            ws1=mmcwatershed(g, mmbinary(mh, 20))
            ws2=mmcwatershed(g, mmbinary(mh, 40))
            mmshow(ws1)
            mmshow(ws2)
    """

    if Bc is None: Bc = mmsecross()
    fplus = mmaddm(f,1)
    g = mmsubm(mmsuprec(fplus,f,Bc),f)
    y = mmunion(mmthreshad(g,1),mmthreshad(f,0,0))
    return y
#
# =====================================================================
#
#   mmse2interval
#
# =====================================================================
def mmse2interval(a, b):
    """
        - Purpose
            Create an interval from a pair of structuring elements.
        - Synopsis
            Iab = mmse2interval(a, b)
        - Input
            a: Structuring Element Left extremity.
            b: Structuring Element Right extremity.
        - Output
            Iab: Interval
        - Description
            mmse2interval creates the interval [a,b] from the structuring
            elements a and b such that a is less or equal b .

    """

    Iab = (a,mmneg(b))
    return Iab
#
# =====================================================================
#
#   mmse2hmt
#
# =====================================================================
def mmse2hmt(A, Bc):
    """
        - Purpose
            Create a Hit-or-Miss Template (or interval) from a pair of
            structuring elements.
        - Synopsis
            Iab = mmse2hmt(A, Bc)
        - Input
            A:  Structuring Element Left extremity.
            Bc: Structuring Element Complement of the right extremity.
        - Output
            Iab: Interval
        - Description
            mmse2hmt creates the Hit-or-Miss Template (HMT), also called
            interval [A,Bc] from the structuring elements A and Bc such that
            A is included in the complement of Bc . The only difference
            between this function and mmse2interval is that here the second
            structuring element is the complement of the one used in the
            other function. The advantage of this function over
            mmse2interval is that this one is more flexible in the use of
            the structuring elements as they are not required to have the
            same size.

    """

    Iab = (A,Bc)
    return Iab
#
# =====================================================================
#
#   mmsebox
#
# =====================================================================
def mmsebox(r=1):
    """
        - Purpose
            Create a box structuring element.
        - Synopsis
            B = mmsebox(r=1)
        - Input
            r: Non-negative integer. Default: 1. Radius.
        - Output
            B: Structuring Element
        - Description
            mmsebox creates the structuring element B formed by r successive
            Minkowski additions of the elementary square (i.e., the 3x3
            square centered at the origin) with itself. If R=0, B is the
            unitary set that contains the origin. If R=1, B is the
            elementary square itself.
        - Examples
            #
            b1 = mmsebox()
            mmseshow(b1)
            b2 = mmsebox(2)
            mmseshow(b2)
    """

    B = mmsesum(mmbinary([[1,1,1],
                          [1,1,1],
                          [1,1,1]]),r)
    return B
#
# =====================================================================
#
#   mmsecross
#
# =====================================================================
def mmsecross(r=1):
    """
        - Purpose
            Diamond structuring element and elementary 3x3 cross.
        - Synopsis
            B = mmsecross(r=1)
        - Input
            r: Double Default: 1. (radius).
        - Output
            B: Structuring Element
        - Description
            mmsecross creates the structuring element B formed by r
            successive Minkowski additions of the elementary cross (i.e.,
            the 3x3 cross centered at the origin) with itself. If r=0, B is
            the unitary set that contains the origin. If r=1 , B is the
            elementary cross itself.
        - Examples
            #
            b1 = mmsecross()
            print mmseshow(b1)
            b2 = mmsecross(2)
            print mmseshow(b2)
    """

    B = mmsesum(mmbinary([[0,1,0],
                          [1,1,1],
                          [0,1,0]]),r)
    return B
#
# =====================================================================
#
#   mmsedisk
#
# =====================================================================
def mmsedisk(r=3, DIM="2D", METRIC="EUCLIDEAN", FLAT="FLAT", h=0):
    """
        - Purpose
            Create a disk or a semi-sphere structuring element.
        - Synopsis
            B = mmsedisk(r=3, DIM="2D", METRIC="EUCLIDEAN", FLAT="FLAT",
            h=0)
        - Input
            r:      Non-negative integer. Default: 3. Disk radius.
            DIM:    String Default: "2D". '1D', '2D, or '3D'.
            METRIC: String Default: "EUCLIDEAN". 'EUCLIDEAN', ' CITY-BLOCK',
                    'OCTAGON', or ' CHESSBOARD'.
            FLAT:   String Default: "FLAT". 'FLAT' or 'NON-FLAT'.
            h:      Double Default: 0. Elevation of the center of the
                    semi-sphere.
        - Output
            B: Structuring Element
        - Description
            mmsedisk creates a flat structuring element B that is disk under
            the metric METRIC , centered at the origin and with radius r or
            a non-flat structuring element that is a semi-sphere under the
            metric METRIC, centered at (0, h) and with radius r . This
            structuring element can be created on the 1D, 2D or 3D space.
        - Examples
            #
            #   example 1
            #
            a=mmseshow(mmsedisk(10,'2D','CITY-BLOCK'))
            b=mmseshow(mmsedisk(10,'2D','EUCLIDEAN'))
            c=mmseshow(mmsedisk(10,'2D','OCTAGON'))
            mmshow(a)
            mmshow(b)
            mmshow(c)
            #
            #   example 2
            #
            d=mmseshow(mmsedisk(10,'2D','CITY-BLOCK','NON-FLAT'))
            e=mmseshow(mmsedisk(10,'2D','EUCLIDEAN','NON-FLAT'))
            f=mmseshow(mmsedisk(10,'2D','OCTAGON','NON-FLAT'))
            mmshow(d)
            mmshow(e)
            mmshow(f)
            #
            #   example 3
            #
            g=mmsedisk(3,'2D','EUCLIDEAN','NON-FLAT')
            mmseshow(g)
            h=mmsedisk(3,'2D','EUCLIDEAN','NON-FLAT',5)
            mmseshow(h)
    """
    from string import upper
    from numpy import resize, transpose, arange
    from numpy import sqrt, arange, transpose, maximum

    METRIC = upper(METRIC)
    FLAT   = upper(FLAT)            
    assert DIM=='2D','Supports only 2D structuring elements'
    if FLAT=='FLAT': y = mmbinary([1])
    else:            y = int32([h])
    if r==0: return y
    if METRIC == 'CITY-BLOCK':
        if FLAT == 'FLAT':
            b = mmsecross(1)
        else:
            b = int32([[-2147483647, 0,-2147483647],
                       [          0, 1,          0],
                       [-2147483647, 0,-2147483647]])
        return mmsedil(y,mmsesum(b,r))
    elif METRIC == 'CHESSBOARD':
        if FLAT == 'FLAT':
            b = mmsebox(1)
        else:
            b = int32([[1,1,1],
                       [1,1,1],
                       [1,1,1]])
        return mmsedil(y,mmsesum(b,r))
    elif METRIC == 'OCTAGON':
        if FLAT == 'FLAT':
            b1,b2 = mmsebox(1),mmsecross(1)
        else:
            b1 = int32([[1,1,1],[1,1,1],[1,1,1]])
            b2 = int32([[-2147483647, 0,-2147483647],
                        [          0, 1,          0],
                        [-2147483647, 0,-2147483647]])
        if r==1: return b1
        else:    return mmsedil( mmsedil(y,mmsesum(b1,r/2)) ,mmsesum(b2,(r+1)/2))
    elif METRIC == 'EUCLIDEAN':
        v = arange(-r,r+1)
        x = resize(v, (len(v), len(v)))
        y = transpose(x)
        Be = mmbinary(sqrt(x*x + y*y) <= (r+0.5))
        if FLAT=='FLAT':
            return Be
        be = h + int32( sqrt( maximum((r+0.5)*(r+0.5) - (x*x) - (y*y),0)))
        be = mmintersec(mmgray(Be,'int32'),be)
        return be
    else:
        assert 0,'Non valid metric'
    return B
#
# =====================================================================
#
#   mmseline
#
# =====================================================================
def mmseline(l=3, theta=0):
    """
        - Purpose
            Create a line structuring element.
        - Synopsis
            B = mmseline(l=3, theta=0)
        - Input
            l:     Non-negative integer. Default: 3.
            theta: Double Default: 0. (degrees, clockwise)
        - Output
            B: Structuring Element
        - Description
            mmseline creates a structuring element B that is a line segment
            that has an extremity at the origin, length l and angle theta (0
            degrees is east direction, clockwise). If l=0 , it generates the
            origin.
        - Examples
            #
            mmseshow(mmseline())
            b1 = mmseline(4,45)
            mmseshow(b1)
            b2 = mmseline(4,-180)
            mmseshow(b2)
            a=mmtext('Line')
            b=mmdil(a,b1)
            mmshow(a)
            mmshow(b)
    """
    from numpy import pi, tan, cos, sin, sign, floor, arange, transpose, array, ones

    theta = pi*theta/180
    if abs(tan(theta)) <= 1:
        s  = sign(cos(theta))
        x0 = arange(0, l * cos(theta)-(s*0.5),s)
        x1 = floor(x0 * tan(theta) + 0.5)
    else:
        s  = sign(sin(theta))
        x1 = arange(0, l * sin(theta) - (s*0.5),s)
        x0 = floor(x1 / tan(theta) + 0.5)
    x = int32(transpose(array([x1,x0])))
    B = mmset2mat((x,mmbinary(ones((x.shape[1],1)))))
    return B
#
# =====================================================================
#
#   mmserot
#
# =====================================================================
def mmserot(B, theta=45, DIRECTION="CLOCKWISE"):
    """
        - Purpose
            Rotate a structuring element.
        - Synopsis
            BROT = mmserot(B, theta=45, DIRECTION="CLOCKWISE")
        - Input
            B:         Structuring Element
            theta:     Double Default: 45. Degrees of rotation. Available
                       values are multiple of 45 degrees.
            DIRECTION: String Default: "CLOCKWISE". 'CLOCKWISE' or '
                       ANTI-CLOCKWISE'.
        - Output
            BROT: Structuring Element
        - Description
            mmserot rotates a structuring element B of an angle theta .
        - Examples
            #
            b = mmimg2se(mmbinary([[0, 0, 0], [0, 1, 1], [0, 0, 0]]));
            mmseshow(b)
            mmseshow(mmserot(b))
            mmseshow(mmserot(b,45,'ANTI-CLOCKWISE'))
    """
    from string import upper
    from numpy import array, sin, cos, transpose
    from numpy import cos, sin, pi, concatenate, transpose, array

    DIRECTION = upper(DIRECTION)            
    if DIRECTION == "ANTI-CLOCKWISE":
       theta = -theta
    SA = mmmat2set(B)
    theta = pi * theta/180
    (y,v)=SA
    if len(y)==0: return mmbinary([0])
    x0 = y[:,1] * cos(theta) - y[:,0] * sin(theta)
    x1 = y[:,1] * sin(theta) + y[:,0] * cos(theta)
    x0 = int32((x0 +0.5)*(x0>=0) + (x0-0.5)*(x0<0))
    x1 = int32((x1 +0.5)*(x1>=0) + (x1-0.5)*(x1<0))
    x = transpose(array([transpose(x1),transpose(x0)]))
    BROT = mmset2mat((x,v))
    return BROT
#
# =====================================================================
#
#   mmseshow
#
# =====================================================================
def mmseshow(B, option="NORMAL"):
    """
        - Purpose
            Display a structuring element as an image.
        - Synopsis
            y = mmseshow(B, option="NORMAL")
        - Input
            B:      Structuring Element
            option: String Default: "NORMAL". 'NORMAL', ' EXPAND' or '
                    NON-FLAT'
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmseshow used with the option EXPAND generates an image y that
            is a suitable graphical representation of the structuring
            element B . This function is useful to convert a structuring
            element to an image. The origin of the structuring element is at
            the center of the image. If B is flat, y is binary, otherwise, y
            is signed int32 image. When using the option NON-FLAT, the
            output y is always a signed int32 image.
        - Examples
            #
            #   example 1
            #
            b=mmsecross(3);
            print mmseshow(b)
            a = mmseshow(b,'EXPAND')
            mmshow(a)
            print mmseshow(b,'NON-FLAT')
            #
            #   example 2
            #
            b=mmsedisk(2,'2D','EUCLIDEAN','NON-FLAT')
            print mmseshow(b)
    """
    from string import upper

    option = upper(option)            
    if option=='NON-FLAT': 
        y=int32([0])
        if mmisbinary(B):
            B = mmintersec(mmgray(B,'int32'),0)
    elif option=='NORMAL':
        if mmisbinary(B):    y=mmbinary([1])
        else:
           y=int32([0])
    elif option=='EXPAND':
        assert mmisbinary(B), 'This option is only available with flat SE'
        y = mmsedil(mmbinary([1]),B)
        b1= mmbinary(y>=0)
        b0= mmero(y,B)
        y = mmbshow(b1,y,b0)
        return y
    else:
        print 'mmseshow: not a valid flag: NORMAL, EXPAND or NON-FLAT'
    y = mmsedil(y,B)
    return y
#
# =====================================================================
#
#   mmsesum
#
# =====================================================================
def mmsesum(B=None, N=1):
    """
        - Purpose
            N-1 iterative Minkowski additions
        - Synopsis
            NB = mmsesum(B=None, N=1)
        - Input
            B: Structuring Element Default: None (3x3 elementary cross).
            N: Non-negative integer. Default: 1.
        - Output
            NB: Structuring Element
        - Description
            mmsesum creates the structuring element NB from N - 1 iterative
            Minkowski additions with the structuring element B .
        - Examples
            #
            #   example 1
            #
            b = mmimg2se(mmbinary([[1, 1, 1], [1, 1, 1], [0, 1, 0]]))
            mmseshow(b)
            b3 = mmsesum(b,3)
            mmseshow(b3)
            #
            #   example 2
            #
            b = mmsedisk(1,'2D','CITY-BLOCK','NON-FLAT');
            mmseshow(b)
            mmseshow(mmsesum(b,2))
    """

    if B is None: B = mmsecross()
    if N==0:
        if mmisbinary(B): return mmbinary([1])
        else:             return int32([0]) # identity
    NB = B
    for i in range(N-1):
        NB = mmsedil(NB,B)
    return NB
#
# =====================================================================
#
#   mmsetrans
#
# =====================================================================
def mmsetrans(Bi, t):
    """
        - Purpose
            Translate a structuring element
        - Synopsis
            Bo = mmsetrans(Bi, t)
        - Input
            Bi: Structuring Element
            t:
        - Output
            Bo: Structuring Element
        - Description
            mmsetrans translates a structuring element by a specific value.
        - Examples
            #
            b1 = mmseline(5)
            mmseshow(b1)
            b2 = mmsetrans(b1,[2,-2])
            mmseshow(b2)
    """

    x,v=mmmat2set(Bi)
    Bo = mmset2mat((x+t,v))
    Bo = Bo.astype(Bi.dtype)
    return Bo
#
# =====================================================================
#
#   mmsereflect
#
# =====================================================================
def mmsereflect(Bi):
    """
        - Purpose
            Reflect a structuring element
        - Synopsis
            Bo = mmsereflect(Bi)
        - Input
            Bi: Structuring Element
        - Output
            Bo: Structuring Element
        - Description
            mmsereflect reflects a structuring element by rotating it 180
            degrees.
        - Examples
            #
            b1 = mmseline(5,30)
            print mmseshow(b1)
            b2 = mmsereflect(b1)
            print mmseshow(b2)
    """

    Bo = mmserot(Bi, 180)
    return Bo
#
# =====================================================================
#
#   mmsedil
#
# =====================================================================
def mmsedil(B1, B2):
    """
        - Purpose
            Dilate one structuring element by another
        - Synopsis
            Bo = mmsedil(B1, B2)
        - Input
            B1: Structuring Element
            B2: Structuring Element
        - Output
            Bo: Structuring Element
        - Description
            mmsedil dilates an structuring element by another. The main
            difference between this dilation and mmdil is that the dilation
            between structuring elements are not bounded, returning another
            structuring element usually larger than anyone of them. This
            gives the composition of the two structuring elements by
            Minkowski addition.
        - Examples
            #
            b1 = mmseline(5)
            mmseshow(b1)
            b2 = mmsedisk(2)
            mmseshow(b2)
            b3 = mmsedil(b1,b2)
            mmseshow(b3)
    """
    from numpy import newaxis, array

    assert ((mmdatatype(B1) == 'binary') or (mmdatatype(B1) == 'int32')) and (
            (mmdatatype(B2) == 'binary') or (mmdatatype(B2) == 'int32')),'SE must be binary or int32'
    if len(B1.shape) == 1: B1 = B1[newaxis,:]
    if len(B2.shape) == 1: B2 = B2[newaxis,:]
    if (mmdatatype(B1) == 'int32') or (mmdatatype(B2) == 'int32'):
       Bo = int32([mmlimits(int32([0]))[0]])
       if mmdatatype(B1) == 'binary':
          B1 = mmgray(B1,'int32',0)
       if mmdatatype(B2) == 'binary':
          B2 = mmgray(B2,'int32',0)
    else:
       Bo = mmbinary([0])
    x,v = mmmat2set(B2)
    if len(x):
        for i in range(x.shape[0]):
            s = mmadd4dil(B1,v[i])
            st= mmsetrans(s,x[i])
            Bo = mmseunion(Bo,st)
    return Bo
#
# =====================================================================
#
#   mmseunion
#
# =====================================================================
def mmseunion(B1, B2):
    """
        - Purpose
            Union of structuring elements
        - Synopsis
            B = mmseunion(B1, B2)
        - Input
            B1: Structuring Element
            B2: Structuring Element
        - Output
            B: Structuring Element
        - Description
            mmseunion creates a structuring element from the union of two
            structuring elements.
        - Examples
            #
            b1 = mmseline(5)
            mmseshow(b1)
            b2 = mmsedisk(3)
            mmseshow(b2)
            b3 = mmseunion(b1,b2)
            mmseshow(b3)
    """
    from numpy import maximum, ones, asarray, newaxis

    assert B1.dtype == B2.dtype, 'Cannot have different datatypes:'
    type1 = B1.dtype
    if len(B1) == 0: return B2
    if len(B1.shape) == 1: B1 = B1[newaxis,:]
    if len(B2.shape) == 1: B2 = B2[newaxis,:]
    if B1.shape <> B2.shape:
        inf = mmlimits(B1)[0]
        h1,w1 = B1.shape
        h2,w2 = B2.shape
        H,W = max(h1,h2),max(w1,w2)
        Hc,Wc = (H-1)/2,(W-1)/2    # center
        BB1,BB2 = asarray(B1),asarray(B2)
        B1, B2  = inf * ones((H,W)), inf *ones((H,W))
        dh1s , dh1e = (h1-1)/2 , (h1-1)/2 + (h1+1)%2 # deal with even and odd dimensions
        dw1s , dw1e = (w1-1)/2 , (w1-1)/2 + (w1+1)%2
        dh2s , dh2e = (h2-1)/2 , (h2-1)/2 + (h2+1)%2
        dw2s , dw2e = (w2-1)/2 , (w2-1)/2 + (w2+1)%2
        B1[ Hc-dh1s : Hc+dh1e+1  ,  Wc-dw1s : Wc+dw1e+1 ] = BB1
        B2[ Hc-dh2s : Hc+dh2e+1  ,  Wc-dw2s : Wc+dw2e+1 ] = BB2
    B = maximum(B1,B2).astype(type1)
    return B
#
# =====================================================================
#
#   mmshow
#
# =====================================================================
def mmshow(f, f1=None, f2=None, f3=None, f4=None, f5=None, f6=None):
    """
        - Purpose
            Display binary or gray-scale images and optionally overlay it
            with binary images.
        - Synopsis
            mmshow(f, f1=None, f2=None, f3=None, f4=None, f5=None, f6=None)
        - Input
            f:  Gray-scale (uint8 or uint16) or binary image.
            f1: Binary image. Default: None. Red overlay.
            f2: Binary image. Default: None. Green overlay.
            f3: Binary image. Default: None. Blue overlay.
            f4: Binary image. Default: None. Magenta overlay.
            f5: Binary image. Default: None. Yellow overlay.
            f6: Binary image. Default: None. Cyan overlay.

        - Description
            Displays the binary or gray-scale (uint8 or uint16) image f ,
            and optionally overlay it with up to six binary images f1 to f6
            in the following colors: f1 as red, f2 as green, f3 as blue, f4
            as yellow, f5 as magenta, and f6 as cian. The image is displayed
            in the MATLAB figure only if no output parameter is given.
        - Examples
            #
            f=mmreadgray('mribrain.tif');
            f150=mmthreshad(f,150);
            f200=mmthreshad(f,200);
            mmshow(f);
            mmshow(f150);
            mmshow(f,f150,f200);
    """
    import adpil

    if len(f.shape) != 2:
       print "Error, mmshow: can only process gray-scale and binary images."
       return
    if   f1 == None: y = mmgshow(f)
    elif f2 == None: y = mmgshow(f,f1)
    elif f3 == None: y = mmgshow(f,f1,f2)
    elif f4 == None: y = mmgshow(f,f1,f2,f3)
    elif f5 == None: y = mmgshow(f,f1,f2,f3,f4)
    elif f6 == None: y = mmgshow(f,f1,f2,f3,f4,f5)
    elif f6 == None: y = mmgshow(f,f1,f2,f3,f4,f5)
    else:            y = mmgshow(f,f1,f2,f3,f4,f5,f6)
    adpil.adshow(y)
    return

#
# =====================================================================
#
#   mmskelm
#
# =====================================================================
def mmskelm(f, B=None, option="binary"):
    """
        - Purpose
            Morphological skeleton (Medial Axis Transform).
        - Synopsis
            y = mmskelm(f, B=None, option="binary")
        - Input
            f:      Binary image.
            B:      Structuring Element Default: None (3x3 elementary
                    cross).
            option: String Default: "binary". Choose one of: binary: output
                    a binary image (medial axis); value: output a grayscale
                    image with values of the radius of the disk to
                    reconstruct the original image (medial axis transform).
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmskelm creates the image y by computing the morphological
            skeleton by B of the image f , when option is BINARY. In this
            case, the pixels of value 1 in y are center of maximal balls
            (generated from B ) included in f . This is also called Medial
            Axis. If option is VALUE, the non zeros pixels in y are the
            radius plus 1 of the maximal balls. This is called Medial Axis
            Transform or valued morphological skeleton.
        - Examples
            #
            #   example 1
            #
            from numpy import ones
            a=mmneg(mmframe(mmbinary(ones((7,9)))))
            print a
            print mmskelm(a)
            print mmskelm(a,mmsebox())
            #
            #   example 2
            #
            a=mmreadgray('pcbholes.tif')
            b=mmskelm(a)
            mmshow(a)
            mmshow(b)
            #
            #   example 3
            #
            c=mmskelm(a,mmsecross(),'value')
            mmshow(c)
    """
    from string import upper
    from numpy import asarray
    if B is None: B = mmsecross()
    assert mmisbinary(f),'Input binary image only'
    option = upper(option)
    k1,k2 = mmlimits(f)
    y = mmgray(mmintersec(f, k1),'uint16')
    iszero = asarray(y)
    nb = mmsesum(B,0)
    for r in range(1,65535):
        ero = mmero( f, nb)
        if mmisequal(ero, iszero): break
        f1 = mmopenth( ero, B)
        nb = mmsedil(nb, B)
        y = mmunion(y, mmgray(f1,'uint16',r))
    if option == 'BINARY':
        y = mmbinary(y)
    return y
#
# =====================================================================
#
#   mmskelmrec
#
# =====================================================================
def mmskelmrec(f, B=None):
    """
        - Purpose
            Morphological skeleton reconstruction (Inverse Medial Axis
            Transform).
        - Synopsis
            y = mmskelmrec(f, B=None)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image.
            B: Structuring Element Default: None (3x3 elementary cross).
        - Output
            y: Binary image.
        - Description
            mmskelmrec reconstructs the valued morphological skeleton to
            recover the original image.
        - Examples
            #
            from numpy import ones
            a=mmneg(mmframe(mmbinary(ones((7,9)))))
            print a
            b=mmskelm(a,mmsecross(),'value')
            print b
            c=mmskelmrec(b,mmsecross())
            print c
    """
    from numpy import ravel
    if B is None: B = mmsecross()
    y = mmbinary(mmintersec(f, 0))
    for r in range(max(ravel(f)),1,-1):
        y = mmdil(mmunion(y,mmbinary(f,r)), B)
    y = mmunion(y, mmbinary(f,1))
    return y
#
# =====================================================================
#
#   mmskiz
#
# =====================================================================
def mmskiz(f, Bc=None, LINEREG="LINES", METRIC=None):
    """
        - Purpose
            Skeleton of Influence Zone - also know as Generalized Voronoi
            Diagram
        - Synopsis
            y = mmskiz(f, Bc=None, LINEREG="LINES", METRIC=None)
        - Input
            f:       Binary image.
            Bc:      Structuring Element Default: None (3x3 elementary
                     cross). Connectivity for the distance measurement.
            LINEREG: String Default: "LINES". 'LINES' or 'REGIONS'.
            METRIC:  String Default: None. 'EUCLIDEAN' if specified.
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmskiz creates the image y by detecting the lines which are
            equidistant to two or more connected components of f , according
            to the connectivity defined by Bc . Depending on with the flag
            LINEREG, y will be a binary image with the skiz lines or a
            labeled image representing the zone of influence regions. When
            the connected objects of f are single points, the skiz is the
            Voronoi diagram.
        - Examples
            #
            #   example 1
            #
            f=mmreadgray('blob2.tif')
            y=mmskiz(f,mmsebox(),'LINES','EUCLIDEAN')
            mmshow(f,y)
            #
            #   example 2
            #
            from numpy import zeros
            f=mmbinary(zeros((100,100)))
            f[30,25],f[20,75],f[50,50],f[70,30],f[80,70] = 1,1,1,1,1
            y = mmskiz(f,mmsebox(),'LINES','EUCLIDEAN')
            mmshow(f,y)
    """
    from string import upper
    if Bc is None: Bc = mmsecross()
    LINEREG = upper(LINEREG)
    if METRIC is not None: METRIC = upper(METRIC)
    d = mmdist( mmneg(f), Bc, METRIC)
    return mmcwatershed(d,f,Bc,LINEREG)
    return y
#
# =====================================================================
#
#   mmstats
#
# =====================================================================
def mmstats(f, measurement):
    """
        - Purpose
            Find global image statistics.
        - Synopsis
            y = mmstats(f, measurement)
        - Input
            f:           
            measurement: String Default: "". Choose the measure to compute:
                         'max', 'min', 'median', 'mean', 'sum', 'std',
                         'std1'.
        - Output
            y:
        - Description
            Compute global image statistics: 'max' - maximum gray-scale
            value in image; 'min' - minimum gray-scale value in image; 'sum'
            - sum of all pixel values; 'median' - median value of all pixels
            in image; 'mean' - mean value of all pixels in image; 'std' -
            standard deviation of all pixels (normalized by N-1); 'std1' -
            idem, normalized by N.

    """
    from string import upper
    from numpy import ravel
    from MLab import mean, median, std

    measurement = upper(measurement)
    if measurement == 'MAX':
        y = max(ravel(f))
    elif measurement == 'MIN':
        y = min(ravel(f))
    elif measurement == 'MEAN':
        y = mean(ravel(f))
    elif measurement == 'MEDIAN':
        y = median(ravel(f))
    elif measurement == 'STD':
        y = std(ravel(f))
    else:
        assert 0,'Not a valid measurement'
    return y
#
# =====================================================================
#
#   mmsubm
#
# =====================================================================
def mmsubm(f1, f2):
    """
        - Purpose
            Subtraction of two images, with saturation.
        - Synopsis
            y = mmsubm(f1, f2)
        - Input
            f1: Unsigned gray-scale (uint8 or uint16), signed (int32) or
                binary image.
            f2: Unsigned gray-scale (uint8 or uint16), signed (int32) or
                binary image. Or constant.
        - Output
            y: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image.
        - Description
            mmsubm creates the image y by pixelwise subtraction of the image
            f2 from the image f1 . When the subtraction of the values of two
            pixels is negative, 0 is taken as the result of the subtraction.
            When f1 and f2 are binary images, y represents the set
            subtraction of f2 from f1 .
        - Examples
            #
            #   example 1
            #
            f = to_uint8([255,   255,    0,   10,   20,   10,    0,   255,  255])
            g = to_uint8([10,     20,   30,   40,   50,   40,   30,    20,    10])
            print mmsubm(f, g)
            print mmsubm(f, 100)
            print mmsubm(100, f)
            #
            #   example 2
            #
            a = mmreadgray('boxdrill-C.tif')
            b = mmreadgray('boxdrill-B.tif')
            c = mmsubm(a,b)
            mmshow(a)
            mmshow(b)
            mmshow(c)
    """
    from numpy import array, clip

    if type(f2) is array:
        assert f1.dtype == f2.dtype, 'Cannot have different datatypes:'
    bottom,top=mmlimits(f1)
    y = clip(f1.astype('d') - f2, bottom, top)
    y = y.astype(f1.dtype)
    return y
#
# =====================================================================
#
#   mmsupcanon
#
# =====================================================================
def mmsupcanon(f, Iab, theta=45, DIRECTION="CLOCKWISE"):
    """
        - Purpose
            Union of sup-generating or hit-miss operators.
        - Synopsis
            y = mmsupcanon(f, Iab, theta=45, DIRECTION="CLOCKWISE")
        - Input
            f:         Binary image.
            Iab:       Interval
            theta:     Double Default: 45. Degrees of rotation: 45, 90, or
                       180.
            DIRECTION: String Default: "CLOCKWISE". 'CLOCKWISE' or '
                       ANTI-CLOCKWISE'
        - Output
            y: Binary image.
        - Description
            mmsupcanon creates the image y by computing the union of
            transformations of the image f by sup-generating operators.
            These hit-miss operators are characterized by rotations (in the
            clockwise or anti-clockwise direction) of theta degrees of the
            interval Iab .

    """
    from string import upper

    DIRECTION = upper(DIRECTION)            
    y = mmintersec(f,0)
    for t in range(0,360,theta):
        Irot = mminterot( Iab, t, DIRECTION )
        y = mmunion( y, mmsupgen(f, Irot))
    return y
#
# =====================================================================
#
#   mmsupgen
#
# =====================================================================
def mmsupgen(f, INTER):
    """
        - Purpose
            Sup-generating (hit-miss).
        - Synopsis
            y = mmsupgen(f, INTER)
        - Input
            f:     Binary image.
            INTER: Interval
        - Output
            y: Binary image.
        - Description
            mmsupgen creates the binary image y by computing the
            transformation of the image f by the sup-generating operator
            characterized by the interval Iab . The sup-generating operator
            is just a relaxed template matching, where the criterion to keep
            a shape is that it be inside the interval Iab . Note that we
            have the classical template matching when a=b . Note yet that
            the sup-generating operator is equivalent to the classical
            hit-miss operator.
        - Examples
            #
            #   example 1
            #
            f=mmbinary([
               [0,0,1,0,0,1,1],
               [0,1,0,0,1,0,0],
               [0,0,0,1,1,0,0]])
            i=mmendpoints()
            print mmintershow(i)
            g=mmsupgen(f,i)
            print g
            #
            #   example 2
            #
            a=mmreadgray('gear.tif')
            b=mmsupgen(a,mmendpoints())
            mmshow(a)
            mmshow(mmdil(b))
    """

    A,Bc = INTER
    y = mmintersec(mmero( f, A),
                   mmero( mmneg(f), Bc))
    return y
#
# =====================================================================
#
#   mmsuprec
#
# =====================================================================
def mmsuprec(f, g, Bc=None):
    """
        - Purpose
            Sup-reconstruction.
        - Synopsis
            y = mmsuprec(f, g, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) or binary image. Marker image.
            g:  Gray-scale (uint8 or uint16) or binary image. Conditioning
                image.
            Bc: Structuring Element Default: None (3x3 elementary cross). (
                connectivity).
        - Output
            y: Image
        - Description
            mmsuprec creates the image y by an infinite number of recursive
            iterations (iterations until stability) of the erosion of f by
            Bc conditioned to g . We say that y is the sup-reconstruction of
            g from the marker f .

    """
    from numpy import product
    if Bc is None: Bc = mmsecross()
    n = product(f.shape)
    y = mmcero(f,g,Bc,n);
    return y
#
# =====================================================================
#
#   mmbshow
#
# =====================================================================
def mmbshow(f1, f2=None, f3=None, factor=17):
    """
        - Purpose
            Generate a graphical representation of overlaid binary images.
        - Synopsis
            y = mmbshow(f1, f2=None, f3=None, factor=17)
        - Input
            f1:     Binary image.
            f2:     Binary image. Default: None.
            f3:     Binary image. Default: None.
            factor: Double Default: 17. Expansion factor for the output
                    image. Use odd values above 9.
        - Output
            y: Binary image. shaded image.
        - Description
            Generate an expanded binary image as a graphical representation
            of up to three binary input images. The 1-pixels of the first
            image are represented by square contours, the pixels of the
            optional second image are represented by circles and for the
            third image they are represented by shaded squares. This
            function is useful to create graphical illustration of small
            images.
        - Examples
            #
            f1=mmtext('b')
            f2=mmtext('w')
            g2=mmbshow(f1,f2)
            mmshow(g2)
            f3=mmtext('x')
            g3=mmbshow(f1,f2,f3)
            mmshow(g3);
    """
    from numpy import newaxis, zeros, resize, transpose, floor, arange, array

    if f1.shape == (): f1 = array([f1])
    if len(f1.shape) == 1: f1 = f1[newaxis,:]
    if (`f1.shape` != `f2.shape`) or \
       (`f1.shape` != `f3.shape`) or \
       (`f2.shape` != `f3.shape`):
        print 'Different sizes.'
        return None
    s = factor
    if factor < 9: s = 9
    h,w = f1.shape
    y = zeros((s*h, s*w))
    xc = resize(range(s), (s,s))
    yc = transpose(xc)
    r  = int(floor((s-8)/2. + 0.5))
    circle = (xc - s/2)**2 + (yc - s/2)**2 <= r**2
    r = arange(s) % 2
    fillrect = resize(array([r, 1-r]), (s,s))
    fillrect[0  ,:] = 0
    fillrect[s-1,:] = 0
    fillrect[:  ,0] = 0
    fillrect[:  ,s-1] = 0
    for i in range(h):
        for j in range(w):
            m, n = s*i, s*j
            if f1 and f1[i,j]:
                y[m     ,n:n+s] = 1
                y[m+s-1 ,n:n+s] = 1
                y[m:m+s ,n    ] = 1
                y[m:m+s ,n+s-1] = 1
            if f2 and f2[i,j]:
                y[m:m+s, n:n+s] = y[m:m+s, n:n+s] + circle
            if f3 and f3[i,j]:
                y[m:m+s, n:n+s] = y[m:m+s, n:n+s] + fillrect
    y = y > 0
    return y
#
# =====================================================================
#
#   mmswatershed
#
# =====================================================================
def mmswatershed(f, g, B=None, LINEREG="LINES"):
    """
        - Purpose
            Detection of similarity-based watershed from markers.
        - Synopsis
            y = mmswatershed(f, g, B=None, LINEREG="LINES")
        - Input
            f:       Gray-scale (uint8 or uint16) image.
            g:       Gray-scale (uint8 or uint16) or binary image. Marker
                     image. If binary, each connected component is an object
                     marker. If gray, it is assumed it is a labeled image.
            B:       Structuring Element Default: None (3x3 elementary
                     cross). (watershed connectivity)
            LINEREG: String Default: "LINES". 'LINES' or ' REGIONS'.
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmswatershed creates the image y by detecting the domain of the
            catchment basins of f indicated by g , according with the
            connectivity defined by B . This watershed is a modified version
            where each basin is defined by a similarity criterion between
            pixels. The original watershed is normally applied to the
            gradient of the image. In this case, the gradient is taken
            internally. According to the flag LINEREG y will be a labeled
            image of the catchment basins domain or just a binary image that
            presents the watershed lines. The implementation of this
            function is based on LotuFalc:00 .
        - Examples
            #
            f = to_uint8([
                [0,  0,  0,  0,  0,  0,  0],
                [0,  1,  0,  0,  0,  1,  0],
                [0,  1,  0,  0,  0,  1,  0],
                [0,  1,  1,  1,  1,  1,  0],
                [0,  1,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0]])
            m = to_uint8([
                [0,  0,  0,  0,  0,  0,  0],
                [0,  1,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  2,  0,  0,  0]])
            print mmswatershed(f,m,mmsecross(),'REGIONS')
    """

    if B is None: B = mmsecross()
    print 'Not implemented yet'
    return None
    return y
#
# =====================================================================
#
#   mmsymdif
#
# =====================================================================
def mmsymdif(f1, f2):
    """
        - Purpose
            Symmetric difference between two images
        - Synopsis
            y = mmsymdif(f1, f2)
        - Input
            f1: Gray-scale (uint8 or uint16) or binary image.
            f2: Gray-scale (uint8 or uint16) or binary image.
        - Output
            y: Image i
        - Description
            mmsymdif creates the image y by taken the union of the
            subtractions of f1 from f2 and f2 from f1 . When f1 and f2 are
            binary images, y represents the set of points that are in f1 and
            not in f2 or that are in f2 and not in f1 .
        - Examples
            #
            #   example 1
            #
            a = to_uint8([1, 2, 3, 4, 5])
            b = to_uint8([5, 4, 3, 2, 1])
            print mmsymdif(a,b)
            #
            #   example 2
            #
            c = mmreadgray('tplayer1.tif')
            d = mmreadgray('tplayer2.tif')
            e = mmsymdif(c,d)
            mmshow(c)
            mmshow(d)
            mmshow(e)
    """

    y = mmunion(mmsubm(f1,f2),mmsubm(f2,f1))
    return y
#
# =====================================================================
#
#   mmtext
#
# =====================================================================
def mmtext(txt):
    """
        - Purpose
            Create a binary image of a text.
        - Synopsis
            y = mmtext(txt)
        - Input
            txt: String Default: "". Text to be written.
        - Output
            y: Binary image.
        - Description
            mmtext creates the binary image y of the text txt . The
            background of y is 0, while its foreground is 1. The text should
            be composed only by lower and upper case letters.

    """
    from numpy import reshape, array

    FontDft = mmbinary([
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   1,   0,   0,
     0,   0,   0,   1,   0,   0,   1,   0,   0,
     0,   0,   0,   1,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   1,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   1,   0,   0,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   1,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   1,   0,   0,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   1,   0,   0,   0,   0,
     0,   1,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   1,   0,
     0,   0,   0,   0,   1,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   1,   0,   0,   1,   0,   0,
     0,   1,   0,   1,   0,   0,   1,   0,   0,
     0,   0,   1,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   1,   0,   0,
     0,   0,   1,   0,   0,   1,   0,   1,   0,
     0,   0,   1,   0,   0,   1,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   1,   0,   0,   0,
     0,   1,   0,   0,   0,   1,   0,   0,   0,
     0,   1,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   1,   1,   1,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   1,   0,   0,   0,
     0,   1,   0,   0,   0,   1,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   1,   1,   0,   0,
     0,   0,   1,   1,   1,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   1,   1,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   1,   1,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   0,   0,   0,   0,
     0,   0,   1,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   1,   1,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   1,   0,   0,
     0,   0,   0,   0,   1,   0,   1,   0,   0,
     0,   0,   0,   1,   0,   0,   1,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   1,   1,   1,   1,   0,   0,
     0,   1,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   1,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   1,   1,   1,   1,   0,   0,
     0,   1,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   1,   0,
     0,   0,   1,   1,   1,   1,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   1,   1,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   1,   1,   1,   0,
     0,   1,   0,   1,   0,   0,   0,   1,   0,
     0,   1,   0,   1,   0,   0,   1,   1,   0,
     0,   1,   0,   0,   1,   1,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   1,   1,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   1,   0,   0,   0,
     0,   1,   0,   0,   1,   0,   0,   0,   0,
     0,   1,   1,   1,   0,   0,   0,   0,   0,
     0,   1,   0,   1,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   1,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   1,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   0,   0,   0,   1,   1,   0,
     0,   1,   0,   1,   0,   1,   0,   1,   0,
     0,   1,   0,   1,   0,   1,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   1,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   1,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   1,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   1,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   1,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   1,   0,   1,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
    55,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   1,   0,
     0,   0,   1,   1,   1,   1,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   1,   1,   1,   1,   0,   0,
     0,   1,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   1,   0,
     0,   0,   1,   1,   1,   1,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   1,   1,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   1,   0,
     0,   0,   0,   1,   0,   0,   0,   1,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   1,   1,   1,   1,   0,   0,
     0,   1,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   1,   1,   0,   0,
     0,   1,   0,   1,   1,   0,   0,   0,   0,
     0,   1,   1,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   1,   1,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   0,   1,   1,   0,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   1,   1,   1,   1,   0,   0,
     0,   1,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   1,   1,   1,   1,   0,   0,
     0,   1,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   1,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   1,   1,   0,
     0,   0,   1,   1,   1,   1,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   1,   1,   1,   0,   0,
     0,   0,   1,   1,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   1,   1,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   0,   1,   0,   0,   1,   0,
     0,   1,   0,   1,   0,   1,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   1,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   1,   1,   0,   0,
     0,   0,   1,   1,   1,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   1,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   1,   1,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   1,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   1,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   1,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   1,   1,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   1,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   1,   1,   1,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   1,   1,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   1,   0,   0,   0,
     0,   0,   0,   0,   1,   1,   0,   0,   0,
     0,   0,   0,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   1,   0,   0,   0,   0,
     0,   1,   1,   1,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0])
    FIRST_CHAR =  32
    LAST_CHAR  = 126
    N_CHARS    = LAST_CHAR - FIRST_CHAR
    WIDTH_DFT  =   9
    HEIGHT_DFT =  15
    FontDft = reshape(FontDft,(HEIGHT_DFT * N_CHARS, WIDTH_DFT))
    txt_i = array(txt,'l') - FIRST_CHAR
    for i in range(len(txt_i)):
      ind = txt_i[i]
      assert ind < (LAST_CHAR-FIRST_CHAR),'mmtext, code not allowed'
      if i == 0:
        y = FontDft[ind*HEIGHT_DFT:(ind+1)*HEIGHT_DFT,:]
      else:
        y = mmconcat('w',y,FontDft[ind*HEIGHT_DFT:(ind+1)*HEIGHT_DFT,:])
    return y
#
# =====================================================================
#
#   mmthick
#
# =====================================================================
def mmthick(f, Iab=None, n=-1, theta=45, DIRECTION="CLOCKWISE"):
    """
        - Purpose
            Image transformation by thickening.
        - Synopsis
            y = mmthick(f, Iab=None, n=-1, theta=45, DIRECTION="CLOCKWISE")
        - Input
            f:         Binary image.
            Iab:       Interval Default: None (mmhomothick).
            n:         Non-negative integer. Default: -1. Number of
                       iterations.
            theta:     Double Default: 45. Degrees of rotation: 45, 90, or
                       180.
            DIRECTION: String Default: "CLOCKWISE". 'CLOCKWISE' or '
                       ANTI-CLOCKWISE'
        - Output
            y: Binary image.
        - Description
            mmthick creates the binary image y by performing a thickening of
            the binary image f . The number of iterations of the thickening
            is n and each iteration is performed by union of f with the
            points that are detected in f by the hit-miss operators
            characterized by rotations of theta degrees of the interval Iab
            .

    """
    from numpy import product
    from string import upper
    if Iab is None: Iab = mmhomothick()
    DIRECTION = upper(DIRECTION)            
    assert mmisbinary(f),'f must be binary image'
    if n == -1: n = product(f.shape)
    y = f
    zero = mmintersec(f,0)
    for i in range(n):
        aux = zero
        for t in range(0,360,theta):
            sup = mmsupgen( y, mminterot(Iab, t, DIRECTION))
            aux = mmunion( aux, sup)
            y = mmunion( y, sup)
        if mmisequal(aux,zero): break
    return y
#
# =====================================================================
#
#   mmthin
#
# =====================================================================
def mmthin(f, Iab=None, n=-1, theta=45, DIRECTION="CLOCKWISE"):
    """
        - Purpose
            Image transformation by thinning.
        - Synopsis
            y = mmthin(f, Iab=None, n=-1, theta=45, DIRECTION="CLOCKWISE")
        - Input
            f:         Binary image.
            Iab:       Interval Default: None (mmhomothin).
            n:         Non-negative integer. Default: -1. Number of
                       iterations.
            theta:     Double Default: 45. Degrees of rotation: 45, 90, or
                       180.
            DIRECTION: String Default: "CLOCKWISE". 'CLOCKWISE' or '
                       ANTI-CLOCKWISE'
        - Output
            y: Binary image.
        - Description
            mmthin creates the binary image y by performing a thinning of
            the binary image f . The number of iterations of the thinning is
            n and each iteration is performed by subtracting the points that
            are detect in f by hit-miss operators characterized by rotations
            of theta of the interval Iab . When n is infinite and the
            interval is mmhomothin (default conditions), mmthin gives the
            skeleton by thinning.
        - Examples
            #
            f=mmreadgray('scissors.tif')
            f1=mmthin(f)
            mmshow(f,f1) # skeleton
            f2=mmthin(f1,mmendpoints(),15) # prunning 15 pixels
            mmshow(f,f2) # prunned skeleton
    """
    from numpy import product
    from string import upper
    if Iab is None: Iab = mmhomothin()
    DIRECTION = upper(DIRECTION)            
    assert mmisbinary(f),'f must be binary image'
    if n == -1: n = product(f.shape)
    y = f
    zero = mmintersec(f,0)
    for i in range(n):
        aux = zero
        for t in range(0,360,theta):
            sup = mmsupgen( y, mminterot(Iab, t, DIRECTION))
            aux = mmunion( aux, sup)
            y = mmsubm( y, sup)
        if mmisequal(aux,zero): break
    return y
#
# =====================================================================
#
#   mmunion
#
# =====================================================================
def mmunion(f1, f2, f3=None, f4=None, f5=None):
    """
        - Purpose
            Union of images.
        - Synopsis
            y = mmunion(f1, f2, f3=None, f4=None, f5=None)
        - Input
            f1: Gray-scale (uint8 or uint16) or binary image.
            f2: Gray-scale (uint8 or uint16) or binary image. Or constant
            f3: Gray-scale (uint8 or uint16) or binary image. Default: None.
                Or constant.
            f4: Gray-scale (uint8 or uint16) or binary image. Default: None.
                Or constant.
            f5: Gray-scale (uint8 or uint16) or binary image. Default: None.
                Or constant.
        - Output
            y: Image
        - Description
            mmunion creates the image y by taking the pixelwise maximum
            between the images f1, f2, f3, f4, and f5 . When f1, f2, f3, f4,
            and f5 are binary images, y represents the union of them.
        - Examples
            #
            #   example 1
            #
            f=to_uint8([255, 255,  0,  10,   0, 255, 250])
            print 'f=',f
            g=to_uint8([  0,  40, 80, 140, 250,  10,  30])
            print 'g=',g
            print mmunion(f, g)
            print mmunion(f, 255)
            #
            #   example 2
            #
            a = mmreadgray('form-ok.tif')
            b = mmreadgray('form-1.tif')
            c = mmunion(a,b)
            mmshow(a)
            mmshow(b)
            mmshow(c)
            #
            #   example 3
            #
            d = mmreadgray('danaus.tif')
            e = mmcmp(d,'<',80)
            f = mmunion(d,mmgray(e))
            mmshow(d)
            mmshow(e)
            mmshow(f)
            #
            #   example 4
            #
            g = mmreadgray('tplayer1.tif')
            h = mmreadgray('tplayer2.tif')
            i = mmreadgray('tplayer3.tif')
            j = mmunion(g,h,i)
            mmshow(g)
            mmshow(h)
            mmshow(i)
            mmshow(j)
    """
    from numpy import maximum

    y = maximum(f1,f2)
    if f3: y = maximum(y,f3)
    if f4: y = maximum(y,f4)
    if f5: y = maximum(y,f5)
    y = y.astype(f1.dtype)
    return y
#
# =====================================================================
#
#   mmversion
#
# =====================================================================
def mmversion():
    """
        - Purpose
            SDC Morphology Toolbox version.
        - Synopsis
            S = mmversion()

        - Output
            S: String ( description of the version).
        - Description
            mmversion gives the SDC Morphology Toolbox version.
        - Examples
            #
            print mmversion()
    """

    return __version_string__
    return S
#
# =====================================================================
#
#   mmwatershed
#
# =====================================================================
def mmwatershed(f, Bc=None, LINEREG="LINES"):
    """
        - Purpose
            Watershed detection.
        - Synopsis
            y = mmwatershed(f, Bc=None, LINEREG="LINES")
        - Input
            f:       Gray-scale (uint8 or uint16) or binary image.
            Bc:      Structuring Element Default: None (3x3 elementary
                     cross). ( connectivity)
            LINEREG: String Default: "LINES". 'LINES' or ' REGIONS'.
        - Output
            y: Gray-scale (uint8 or uint16) or binary image.
        - Description
            mmwatershed creates the image y by detecting the domain of the
            catchment basins of f , according to the connectivity defined by
            Bc . According to the flag LINEREG y will be a labeled image of
            the catchment basins domain or just a binary image that presents
            the watershed lines. The implementation of this function is
            based on VincSoil:91 .
        - Examples
            #
            f=mmreadgray('astablet.tif')
            grad=mmgradm(f)
            w1=mmwatershed(grad,mmsebox())
            w2=mmwatershed(grad,mmsebox(),'REGIONS')
            mmshow(grad)
            mmshow(w1)
            mmlblshow(w2)
    """
    from string import upper
    if Bc is None: Bc = mmsecross()
    return mmcwatershed(f,mmregmin(f,Bc),upper(LINEREG))
    return y
#
# =====================================================================
#
#   mmbench
#
# =====================================================================
def mmbench(count=10):
    """
        - Purpose
            benchmarking main functions of the toolbox.
        - Synopsis
            mmbench(count=10)
        - Input
            count: Double Default: 10. Number of repetitions of each
                   function.

        - Description
            mmbench measures the speed of many of SDC Morphology Toolbox
            functions in seconds. An illustrative example of the output of
            mmbench is, for a MS-Windows 2000 Pentium 4, 2.4GHz, 533MHz
            system bus, machine: SDC Morphology Toolbox V1.2 27Sep02
            Benchmark Made on Wed Jul 16 15:33:17 2003 computer= win32 image
            filename= csample.jpg width= 640 , height= 480 Function time
            (sec.) 1. Union bin 0.00939999818802 2. Union gray-scale
            0.00319999456406 3. Dilation bin, mmsecross 0.0110000014305 4.
            Dilation gray, mmsecross 0.00780000686646 5. Dilation gray,
            non-flat 3x3 SE 0.0125 6. Open bin, mmsecross 0.0125 7. Open
            gray-scale, mmsecross 0.0141000032425 8. Open gray, non-flat 3x3
            SE 0.0235000014305 9. Distance mmsecross 0.021899998188 10.
            Distance Euclidean 0.0264999985695 11. Geodesic distance
            mmsecross 0.028100001812 12. Geodesic distance Euclidean
            0.303100001812 13. Area open bin 0.0639999985695 14. Area open
            gray-scale 0.148500001431 15. Label mmsecross 0.071899998188 16.
            Regional maximum, mmsecross 0.043700003624 17. Open by rec,
            gray, mmsecross 0.0515000104904 18. ASF by rec, oc, mmsecross, 1
            0.090600001812 19. Gradient, gray-scale, mmsecross
            0.0171999931335 20. Thinning 0.0984999895096 21. Watershed
            0.268799996376 Average 0.0632523809161

    """
    from sys import platform
    from time import time, asctime
    from numpy import average, zeros

    filename = 'csample.jpg'
    f = mmreadgray(filename)
    fbin=mmthreshad(f,150)
    se = mmimg2se(mmbinary([[0,1,0],[1,1,1],[0,1,0]]),'NON-FLAT',int32([[0,1,0],[1,2,1],[0,1,0]]))
    m=mmthin(fbin)
    tasks=[
       [' 1. Union  bin                      ','mmunion(fbin,fbin)'],
       [' 2. Union  gray-scale               ','mmunion(f,f)'],
       [' 3. Dilation  bin, mmsecross        ','mmdil(fbin)'],
       [' 4. Dilation  gray, mmsecross       ','mmdil(f)'],
       [' 5. Dilation  gray, non-flat 3x3 SE ','mmdil(f,se)'],
       [' 6. Open      bin, mmsecross        ','mmopen(fbin)'],
       [' 7. Open      gray-scale, mmsecross ','mmopen(f)'],
       [' 8. Open      gray, non-flat 3x3 SE ','mmopen(f,se)'],
       [' 9. Distance  mmsecross             ','mmdist(fbin)'],
       ['10. Distance  Euclidean             ','mmdist(fbin,mmsebox(),"euclidean")'],
       ['11. Geodesic distance mmsecross     ','mmgdist(fbin,m)'],
       ['12. Geodesic distance Euclidean     ','mmgdist(fbin,m,mmsebox(),"euclidean")'],
       ['13. Area open bin                   ','mmareaopen(fbin,100)'],
       ['14. Area open gray-scale            ','mmareaopen(f,100)'],
       ['15. Label mmsecross                 ','mmlabel(fbin)'],
       ['16. Regional maximum, mmsecross     ','mmregmax(f)'],
       ['17. Open by rec, gray, mmsecross    ','mmopenrec(f)'],
       ['18. ASF by rec, oc, mmsecross, 1    ','mmasfrec(f)'],
       ['19. Gradient, gray-scale, mmsecross ','mmgradm(f)'],
       ['20. Thinning                        ','mmthin(fbin)'],
       ['21. Watershed                       ','mmcwatershed(f,fbin)']]
    result = zeros((21),'d')
    for t in range(len(tasks)):
       print tasks[t][0],tasks[t][1]
       t1=time()
       for k in range(count):
          a=eval(tasks[t][1])
       t2=time()
       result[t]= (t2-t1)/(count+0.0)
    print mmversion() +' Benchmark'
    print 'Made on ',asctime(),' computer=',platform
    print 'image filename=',filename,' width=', f.shape[1],', height=',f.shape[0]
    print '    Function                            time (sec.)'
    for j in range(21):
     print tasks[j][0], result[j]
    print '    Average                         ', average(result)
    out=[]

#
# =====================================================================
#
#   mminstall
#
# =====================================================================
def mminstall(code=None):
    """
        - Purpose
            Verify if the Morphology Toolbox is registered.
        - Synopsis
            mminstall(code=None)
        - Input
            code: String Default: None. Authorization code.

        - Description
            mminstall verifies if the toolbox is registered or not. If not,
            it identifies the internal code that must be used to get the
            authorization code from the software manufacturer.

    """

    mmver = ''
    if code is None:
       s = mmregister()
       if s[0:8] != 'Licensed':
          print 'Please access the web site http://www.mmorph.com/cgi-bin/pymorph-reg.cgi'
          print 'and use the internal code below to obtain your license'
          print s
          print 'If you have any difficulty, please inform morph@mmorph.com'
          return
    else:
       if mmregister(code,'pymorph_license.txt'):
          s=mmregister()
          if s[0:8] != 'Licensed':
            print 'Could not license the toolbox.'
            return
          else:
            print 'Toolbox registered successfully.'
       else:
          print 'The library file could not be registered'
          return

#
# =====================================================================
#
#   mmmaxleveltype
#
# =====================================================================
def mmmaxleveltype(TYPE='uint8'):
    """
        - Purpose
            Returns the maximum value associated to an image datatype
        - Synopsis
            max = mmmaxleveltype(TYPE='uint8')
        - Input
            TYPE: String Default: 'uint8'. One of the strings 'uint8',
                  'uint16' or 'int32', specifying the image type
        - Output
            max: the maximum level value of type TYPE

    """

    max = 0
    if   TYPE == 'uint8'  : max=255
    elif TYPE == 'binary' : max=1
    elif TYPE == 'uint16' : max=65535
    elif TYPE == 'int32'  : max=2147483647
    else:
        assert 0, 'does not support this data type:'+TYPE
    return max
#
# =====================================================================
#
#   int32
#
# =====================================================================
def int32(f):
    """
        - Purpose
            Convert an image to an int32 image.
        - Synopsis
            img = int32(f)
        - Input
            f: Any image
        - Output
            img: The converted image
        - Description
            int32 clips the input image between the values -2147483647 and
            2147483647 and converts it to the signed 32-bit datatype.

    """
    from numpy import array, clip

    img = array(clip(f,-2147483647,2147483647)).astype('i')
    return img
#
# =====================================================================
#
#   uint8
#
# =====================================================================
def to_uint8(f):
    """
        - Purpose
            Convert an image to an uint8 image.
        - Synopsis
            img = to_uint8(f)
        - Input
            f: Any image
        - Output
            img: Gray-scale uint8 image. The converted image
        - Description
            uint8 clips the input image between the values 0 and 255 and
            converts it to the unsigned 8-bit datatype.
        - Examples
            #
            a = int32([-3,0,8,600])
            print to_uint8(a)
    """
    from numpy import array, clip, uint8

    img = array(clip(f,0,255),uint8)
    return img
#
# =====================================================================
#
#   uint16
#
# =====================================================================
def uint16(f):
    """
        - Purpose
            Convert an image to a uint16 image.
        - Synopsis
            img = uint16(f)
        - Input
            f: Any image
        - Output
            img: The converted image
        - Description
            uint16 clips the input image between the values 0 and 65535 and
            converts it to the unsigned 16-bit datatype.
        - Examples
            #
            a = int32([-3,0,8,100000])
            print uint16(a)
    """
    from numpy import array, clip

    img = array(clip(f,0,65535)).astype('w')
    return img
#
# =====================================================================
#
#   mmdatatype
#
# =====================================================================
def mmdatatype(f):
    """
        - Purpose
            Return the image datatype string
        - Synopsis
            type = mmdatatype(f)
        - Input
            f: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image. Any image
        - Output
            type: String String representation of image type: 'binary',
                  'uint8', 'uint16' or 'int32'
        - Description
            mmdatatype returns a string that identifies the pixel datatype
            of the image f .

    """

    code = f.dtype
    if   code == bool: type='binary'
    elif code == uint8: type='uint8'
    elif code == uint16: type='uint16'
    elif code == int32: type='int32'
    else:
        assert 0,'Does not accept this typecode:'+code
    return type
#
# =====================================================================
#
#   mmadd4dil
#
# =====================================================================
def mmadd4dil(f, c):
    """
        - Purpose
            Addition for dilation
        - Synopsis
            a = mmadd4dil(f, c)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image. Image
            c: Gray-scale (uint8 or uint16) or binary image. Constant
        - Output
            a: Image f + c

    """
    from numpy import asarray, minimum, maximum

    if c:            
       y = asarray(f,'d') + c
       k1,k2 = mmlimits(f)
       y = ((f==k1) * k1) + ((f!=k1) * y)
       y = maximum(minimum(y,k2),k1)
       a = y.astype(f.dtype)
    else:
       a = f
    return a
#
# =====================================================================
#
#   mmmat2set
#
# =====================================================================
def mmmat2set(A):
    """
        - Purpose
            Converts image representation from matrix to set
        - Synopsis
            CV = mmmat2set(A)
        - Input
            A: Image in matrix format, where the origin (0,0) is at the
               center of the matrix.
        - Output
            CV: Image Tuple with array of pixel coordinates and array of
                corresponding pixel values
        - Description
            Return tuple with array of pixel coordinates and array of
            corresponding pixel values. The input image is in the matrix
            format, like the structuring element, where the origin (0,0) is
            at the center of the matrix.
        - Examples
            #
            #   example 1
            #
            f=to_uint8([[1,2,3],[4,5,6],[7,8,9]])
            i,v=mmmat2set(f)
            print i
            print v
            #
            #   example 2
            #
            f=to_uint8([[1,2,3,4],[5,6,7,8]])
            i,v=mmmat2set(f)
            print i
            print v
    """
    from numpy import take, ravel, nonzero, transpose, newaxis

    if len(A.shape) == 1: A = A[newaxis,:]
    offsets = nonzero(ravel(A) - mmlimits(A)[0])[0]
    if len(offsets) == 0: return ([],[])
    (h,w) = A.shape
    x = range(2)
    x[0] = offsets/w - (h-1)/2
    x[1] = offsets%w - (w-1)/2
    x = transpose(x)
    CV = x,take(ravel(A),offsets)
    return CV
#
# =====================================================================
#
#   mmset2mat
#
# =====================================================================
def mmset2mat(A):
    """
        - Purpose
            Converts image representation from set to matrix
        - Synopsis
            M = mmset2mat(A)
        - Input
            A: Tuple with array of pixel coordinates and optional array of
               corresponding pixel values
        - Output
            M: Image in matrix format, origin (0,0) at the matrix center
        - Description
            Return an image in the matrix format built from a tuple of an
            array of pixel coordinates and a corresponding array of pixel
            values
        - Examples
            #
            coord=int32([
              [ 0,0],
              [-1,0],
              [ 1,1]])
            A=mmset2mat((coord,))
            print A
            print mmdatatype(A)
            vu = to_uint8([1,2,3])
            f=mmset2mat((coord,vu))
            print f
            print mmdatatype(f)
            vi = int32([1,2,3])
            g=mmset2mat((coord,vi))
            print g
            print mmdatatype(g)
    """
    from MLab import max
    from numpy import put, ones, ravel, shape, newaxis, array, asarray

    if len(A) == 2:            
        x, v = A
        v = asarray(v)
    elif len(A) == 1:
        x = A[0]
        v = ones((len(x),), '1')
    else:
        raise TypeError, 'Argument must be a tuple of length 1 or 2'
    if len(x) == 0:  return array([0]).astype(v.dtype)
    if len(x.shape) == 1: x = x[newaxis,:]
    dh,dw = max(abs(x))
    h,w = (2*dh)+1, (2*dw)+1 
    M=ones((h,w)) * mmlimits(v)[0]
    offset = x[:,0] * w + x[:,1] + (dh*w + dw)
    put(M,offset,v)
    M = M.astype(v.dtype)
    return M
#
# =====================================================================
#
#   mmpad4n
#
# =====================================================================
def mmpad4n(f, Bc, value, scale=1):
    """
        - Purpose
            mmpad4n
        - Synopsis
            y = mmpad4n(f, Bc, value, scale=1)
        - Input
            f:     Image
            Bc:    Structuring Element ( connectivity).
            value: 
            scale: Default: 1.
        - Output
            y: The converted image

    """
    from numpy import ones, array

    if type(Bc) is not array:
      Bc = mmseshow(Bc)            
    Bh, Bw = Bc.shape
    assert Bh%2 and Bw%2, 'structuring element must be odd sized'
    ch, cw = scale * Bh/2, scale * Bw/2
    g = value * ones( f.shape + scale * (array(Bc.shape) - 1))
    g[ ch: -ch, cw: -cw] = f
    y = g.astype(f.dtype)
    return y
#
# =====================================================================
#
#   Global statements for mmplot
#
# =====================================================================
__figs__ = [None]
# =====================================================================
#
#   mmplot
#
# =====================================================================
def mmplot(plotitems=[], options=[], outfig=-1, filename=None):
    """
        - Purpose
            Plot a function.
        - Synopsis
            fig = mmplot(plotitems=[], options=[], outfig=-1, filename=None)
        - Input
            plotitems: Default: []. List of plotitems.
            options:   Default: []. List of options.
            outfig:    Default: -1. Integer. Figure number. 0 creates a new
                       figure.
            filename:  Default: None. String. Name of the PNG output file.
        - Output
            fig: Figure number.

        - Examples
            #
            import numpy
            #
            x = numpy.arange(0, 2*numpy.pi, 0.1)
            mmplot([[x]])
            y1 = numpy.sin(x)
            y2 = numpy.cos(x)
            opts = [['title', 'Example Plot'],\
                    ['grid'],\
                    ['style', 'linespoints'],\
                    ['xlabel', '"X values"'],\
                    ['ylabel', '"Y Values"']]
            y1_plt = [x, y1, None,    'sin(X)']
            y2_plt = [x, y2, 'lines', 'cos(X)']
            #
            # plotting two graphs using one step
            fig1 = mmplot([y1_plt, y2_plt], opts, 0)
            #
            # plotting the same graphs using two steps
            fig2 = mmplot([y1_plt], opts, 0)
            fig2 = mmplot([y2_plt], opts, fig2)
            #
            # first function has been lost, lets recover it
            opts.append(['replot'])
            fig2 = mmplot([y1_plt], opts, fig2)
    """
    import Gnuplot
    import numpy

    newfig = 0
    if (plotitems == 'reset'):
        __figs__[0] = None
        __figs__[1:] = []
        return 0
    if len(plotitems) == 0:
        # no plotitems specified: replot current figure
        if __figs__[0]:
            outfig = __figs__[0]
            g = __figs__[outfig]
            g.replot()
            return outfig
        else:
            #assert 0, "mmplot error: There is no current figure\n"
            print "mmplot error: There is no current figure\n"
            return 0
    # figure to be plotted
    if ((outfig < 0) and __figs__[0]):
        # current figure
        outfig = __figs__[0]
    elif ( (outfig == 0) or ( (outfig == -1) and not __figs__[0] )  ):
        # new figure
        newfig = 1
        outfig = len(__figs__)
    elif outfig >= len(__figs__):
        #assert 0, 'mmplot error: Figure ' + str(outfig) + 'does not exist\n'
        print 'mmplot error: Figure ' + str(outfig) + 'does not exist\n'
        return 0
    #current figure
    __figs__[0] = outfig
    # Gnuplot pointer
    if newfig:
        if len(__figs__) > 20:
            print '''mmplot error: could not create figure. Too many PlotItems in memory (20). Use
                     mmplot('reset') to clear table'''
            return 0

        g = Gnuplot.Gnuplot()
        __figs__.append(g)
    else:
        g = __figs__[outfig]

    # options
    try:
        options.remove(['replot'])
    except:
        g.reset()
    try:
        #default style
        g('set data style lines')
        for option in options:
            if option[0] == 'grid':
                g('set grid')
            elif option[0] == 'title':
                g('set title "' + option[1] + '"')
            elif option[0] == 'xlabel':
                g('set xlabel ' + option[1])
            elif option[0] == 'ylabel':
                g('set ylabel ' + option[1])
            elif option[0] == 'style':
                g('set data style ' + option[1])
            else:
                print "mmplot warning: Unknown option: " + option[0]
    except:
        print "mmplot warning: Bad usage in options! Using default values. Please, use help.\n"
    # Plot items: item[0]=x, item[1]=y, item[2]=style
    for item in plotitems:
        try:
            title = None
            style = None
            x = numpy.ravel(item[0])
            if len(item) > 1:
                # y axis specified
                y = numpy.ravel(item[1])
                if len(item) > 2:
                    # style specified
                    style = item[2]
                    if len(item) > 3:
                        title = item[3]
            else:
                # no y axis specified
                y = x
                x = numpy.arange(len(y))
            g.replot(Gnuplot.Data(x, y, title=title, with=style))
        except:
            g.reset()
            if newfig:
                __figs__.pop()
            #assert 0, "mmplot error: Bad usage in plotitems! Impossible to plot graph. Please, use help.\n"
            print "mmplot error: Bad usage in plotitems! Impossible to plot graph. Please, use help.\n"
            return 0
    # PNG file
    if filename:
        g.hardcopy(filename, terminal='png', color=1)
    fig = outfig
    return fig
#
#
#
# =====================================================================
#  Adesso -- Generated Mon Aug 04 12:09:00 Hora oficial do Brasil 2003
# =====================================================================
#

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
