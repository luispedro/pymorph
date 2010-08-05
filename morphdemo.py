"""
    Module morphdemo -- Demonstrations
    -------------------------------------------------------------------
    morphdemo is a set of Demonstrations for pymorph package
    -------------------------------------------------------------------
    airport()     -- Detecting runways in satellite airport imagery.
    area()        -- Remove objects with small areas in binary images.
    asp()         -- Detect the missing aspirin tablets in a card of aspirin
                     tablets.
    beef()        -- Detect the lean meat region in a beef steak image.
    blob()        -- Demonstrate blob measurements and display.
    brain()       -- Extract the lateral ventricle from an MRI image of the
                     brain.
    calc()        -- Extract the keys of a calculator.
    cells()       -- Extract blood cells and separate them.
    chickparts()  -- Classify chicken parts in breast, legs+tights and wings
    concrete()    -- Aggregate and anhydrous phase extraction from a concrete
                     section observed by a SEM image.
    cookies()     -- Detect broken rounded biscuits.
    cornea()      -- Cornea cells marking.
    fabric()      -- Detection of vertical weave in fabrics.
    fila()        -- Detect Filarial Worms.
    flatzone()    -- Flat-zone image simplification by connected filtering.
    flow()        -- Detect water in a static image of an oil-water flow
                     experiment.
    gear()        -- Detect the teeth of a gear
    holecenter()  -- Hole center misalignment in PCB.
    labeltext()   -- Segmenting letters, words and paragraphs.
    leaf()        -- Segment a leaf from the background
    lith()        -- Detect defects in a microelectronic circuit.
    pcb()         -- Decompose a printed circuit board in its main parts.
    pieces()      -- Classify two dimensional pieces.
    potatoes()    -- Grade potato quality by shape and skin spots.
    robotop()     -- Detect marks on a robot.
    ruler()       -- Detect defects in a ruler.
    soil()        -- Detect fractures in soil.
"""
from pymorph import *
import numpy

print '''\
*********************** WARNING ******************************
The demo is not as well maintained as the rest of the package.
*********************** WARNING ******************************

The demo has not been updated to the newer interfaces.
'''

def readgray(imgname):
    import pylab
    return pylab.imread('pymorph/data/' + imgname)

def show(f, f1=None, f2=None, f3=None, f4=None, f5=None, f6=None):
    import pylab
    pylab.ion()
    pylab.imshow(overlay(f,f1,f2,f3,f4,f5,f6))
    pylab.draw()


# =========================================================================
#
#   airport - Detecting runways in satellite airport imagery.
#
# =========================================================================
def airport():

    print
    print '''Detecting runways in satellite airport imagery.'''
    print
    #
    print '========================================================================='
    print '''
    The satellite image of the airport is read.
    '''
    print '========================================================================='
    #0
    print '''
    f = readgray('galeao.jpg')
    show(f)'''
    f = readgray('galeao.jpg')
    show(f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The disk of radius 5 (diameter 11) is chosen to detect features
    smaller than this size. For visualization, the top-hat image is
    brightened by 150 gray-levels.
    '''
    print '========================================================================='
    #0
    print '''
    th=openth(f,sedisk(5))
    show(addm(th, 150))'''
    th=openth(f,sedisk(5))
    show(addm(th, 150))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A thresholding is applied to detect the features enhanced by the
    top-hat. This is a standard top-hat sequence.
    '''
    print '========================================================================='
    #0
    print '''
    bin=threshad(th,30)
    show(f,bin)'''
    bin=threshad(th,30)
    show(f,bin)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The thinning (red) and pruning (green) detect closed structures
    which characterized the runways structure. The area open (blue)
    selects only very long features, with more than 1000 pixels.
    '''
    print '========================================================================='
    #0
    print '''
    m1=thin(bin)
    m2=thin(m1,endpoints())
    m=areaopen(m2,1000,sebox())
    show(f,m1,m2,m)'''
    m1=thin(bin)
    m2=thin(m1,endpoints())
    m=areaopen(m2,1000,sebox())
    show(f,m1,m2,m)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The previous result is a sample of the runway pixels. It is used as
    a marker for gray-scale morphological reconstruction. The runways
    are enhanced in the reconstructed image.
    '''
    print '========================================================================='
    #0
    print '''
    g=infrec(gray(m), th)
    show(g)'''
    g=infrec(gray(m), th)
    show(g)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A thresholding is applied to the reconstructed image, detecting the
    airport runways.
    '''
    print '========================================================================='
    #0
    print '''
    final=threshad(g, 20)
    show(f, final)'''
    final=threshad(g, 20)
    show(f, final)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   area - Remove objects with small areas in binary images.
#
# =========================================================================
def area():

    print
    print '''Remove objects with small areas in binary images.'''
    print
    #
    print '========================================================================='
    print '''
    The binary image to be processed is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('circuit_bw.tif')
    show(a)'''
    a = readgray('circuit_bw.tif')
    show(a)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The procedure areaopen removes the objects with area less than the
    specified parameter (i.e., 200).
    '''
    print '========================================================================='
    #0
    print '''
    b = areaopen(a,200)
    show(b)'''
    b = areaopen(a,200)
    show(b)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    For displaying purposes the filtered image is superposed over the
    original image.
    '''
    print '========================================================================='
    #0
    print '''
    show(a,b)'''
    show(a,b)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   asp - Detect the missing aspirin tablets in a card of aspirin tablets.
#
# =========================================================================
def asp():

    print
    print '''Detect the missing aspirin tablets in a card of aspirin tablets.'''
    print
    #
    print '========================================================================='
    print '''
    The aspirin tablet binary image is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('astablet.tif')
    show(a)'''
    a = readgray('astablet.tif')
    show(a)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The image can be model as a topographical surface where white
    regions corresponds to high altitude and dark regions to lower
    altitute. The regional maxima of the image is normally very noisy as
    can be seen below.
    '''
    print '========================================================================='
    #0
    print '''
    b = surf(a)
    show(b)
    c = regmax(a,sebox())
    show(b,c)'''
    b = surf(a)
    show(b)
    c = regmax(a,sebox())
    show(b,c)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Opening the original image by a disk a little smaller than the
    tablets removes all the noisy regional maxima. The only regional
    maxima in the opened image are the aspirin tablets as they are the
    only regionally brighter regions of shape larger than the disk of
    radius 20 pixels.
    '''
    print '========================================================================='
    #0
    print '''
    d = open(a, sedisk(20))
    e = surf(d)
    show(e)
    f = regmax(d,sebox())
    show(e,f)'''
    d = open(a, sedisk(20))
    e = surf(d)
    show(e)
    f = regmax(d,sebox())
    show(e,f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Here it is shown the input and output result. Note that the binary
    image of the aspirin tablets was obtained using just one parameter:
    the radius of the circular structuring element. The problem was
    solved as treating the image formed by circular bright regions.
    '''
    print '========================================================================='
    #0
    print '''
    show(a)
    show(f)'''
    show(a)
    show(f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   labeltext - Segmenting letters, words and paragraphs.
#
# =========================================================================
def labeltext():

    print
    print '''Segmenting letters, words and paragraphs.'''
    print
    #
    print '========================================================================='
    print '''
    The text image is read.
    '''
    print '========================================================================='
    #0
    print '''
    f = readgray('stext.tif')
    show(f)'''
    f = readgray('stext.tif')
    show(f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The letters are the main connected components in the image. So we
    use the classical 8-connectivity criteria for identify each letter.
    '''
    print '========================================================================='
    #0
    print '''
    fl=label(f,sebox())
    lblshow(fl)'''
    fl=label(f,sebox())
    lblshow(fl)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The words are made of closed letters. In this case we use a
    connectivity specified by a rectangle structuring element of 7
    pixels high and 11 pixels width, so any two pixels that can be hit
    by this rectangle, belong to the same connected component. The
    values 7 and 11 were chosen experimentally and depend on the font
    size.
    '''
    print '========================================================================='
    #0
    print '''
    from numpy.oldnumeric import ones
    sew = img2se(binary(ones((7,11))))
    seshow(sew)
    fw=label(f,sew)
    lblshow(fw)'''
    from numpy.oldnumeric import ones
    sew = img2se(binary(ones((7,11))))
    seshow(sew)
    fw=label(f,sew)
    lblshow(fw)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Similarly, paragraphs are closed words. In this case the
    connectivity is given by a rectangle of 35 by 20 pixels.
    '''
    print '========================================================================='
    #0
    print '''
    sep = img2se(binary(ones((20,35))))
    fp=label(f,sep)
    lblshow(fp)'''
    sep = img2se(binary(ones((20,35))))
    fp=label(f,sep)
    lblshow(fp)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   beef - Detect the lean meat region in a beef steak image.
#
# =========================================================================
def beef():

    print
    print '''Detect the lean meat region in a beef steak image.'''
    print
    #
    print '========================================================================='
    print '''
    The gray-scale image of the beef steak is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('beef.tif');
    show(a);'''
    a = readgray('beef.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The input image is simplified by the application of a a small
    closing. The dark area (inner lean part) is closed from the fat
    white area.
    '''
    print '========================================================================='
    #0
    print '''
    b=close(a,sedisk(2));
    show(b);'''
    b=close(a,sedisk(2));
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The external marker is built from the detection of the complete beef
    region and the extraction of its internal edge. As the beef is dark,
    it is detected by a low value threshold. After this threshold, small
    residual regions are eliminated by the binary areaclose operator.
    '''
    print '========================================================================='
    #0
    print '''
    c = threshad(a,uint8(10));
    d = areaclose(c,200);
    show(d);'''
    c = threshad(a,uint8(10));
    d = areaclose(c,200);
    show(d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The internal edge generated is 13 points thick. It is created by the
    residues of an erosion by a large structuring element.
    '''
    print '========================================================================='
    #0
    print '''
    e = gradm(d,secross(1),sebox(13));
    show(e);'''
    e = gradm(d,secross(1),sebox(13));
    show(e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The internal marker is a severe erosion of the steak. Both markers
    are combined by union and displayed as overlay on the gradient image
    '''
    print '========================================================================='
    #0
    print '''
    f= erode(d,secross(80));
    g = union(e,f);
    h = gradm(b);
    show(h,g);'''
    f= erode(d,secross(80));
    g = union(e,f);
    h = gradm(b);
    show(h,g);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Constrained watershed of the gradient of the smoothed image,
    restricted to the internal and external markers
    '''
    print '========================================================================='
    #0
    print '''
    i=cwatershed(h,g);
    show(i);'''
    i=cwatershed(h,g);
    show(i);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Superposition of the dilated detected contour on the original image.
    '''
    print '========================================================================='
    #0
    print '''
    show(a,dilate(i));'''
    show(a,dilate(i));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   blob - Demonstrate blob measurements and display.
#
# =========================================================================
def blob():

    print
    print '''Demonstrate blob measurements and display.'''
    print
    #
    print '========================================================================='
    print '''
    The binary image is read and then labeled. The number of blobs is
    measured as the maximum label value. Both images are displayed.
    '''
    print '========================================================================='
    #0
    print '''
    f  = readgray('blob3.tif')
    fr = label(f)
    show(f)
    lblshow(fr,'border')
    nblobs=stats(fr,'max')
    print nblobs'''
    f  = readgray('blob3.tif')
    fr = label(f)
    show(f)
    lblshow(fr,'border')
    nblobs=stats(fr,'max')
    print nblobs
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The centroids are computed from the labeled image. After, the
    centroid image is labeled, so that each centroid point has a label
    value varying from 1 to the maximum number of blobs. For display
    illustration, the centroids are overlayed on the original blob image
    on the left and the labeled centroids are enlarged and displayed on
    the right.
    '''
    print '========================================================================='
    #0
    print '''
    c  = blob(fr,'centroid')
    cr = label(c)
    show(f,c)
    lblshow(dilate(cr))'''
    c  = blob(fr,'centroid')
    cr = label(c)
    show(f,c)
    lblshow(dilate(cr))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    To place a particular number on a particular blob, a number image is
    generated using the function and converted to a structuring element.
    A particular centroid is selected by comparing the image with the
    labeled number. This output image is a binary image with a single
    point at that centroid. Dilating this image by the structuring
    element will "stamp" the structuring element on the centroid.
    '''
    print '========================================================================='
    #0
    print '''
    fbin = cmp(cr,'==',uint16(5))
    f5   = text('5')
    print f5
    b5   = img2se(f5)
    fb5  = dilate(fbin,b5)
    show(dilate(fbin))
    show(f,fb5)'''
    fbin = cmp(cr,'==',uint16(5))
    f5   = text('5')
    print f5
    b5   = img2se(f5)
    fb5  = dilate(fbin,b5)
    show(dilate(fbin))
    show(f,fb5)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    To automate the process just described, a loop scans every label
    value and "stamp" its number in a final image. The stamps are
    accumulated with the function. The area is computed and plotted
    against each label blob number.
    '''
    print '========================================================================='
    #0
    print '''
    facc=subm(f,f)
    for i in range(1,nblobs+1):
      fbin = cmp(cr,'==',uint16(i))
      fi   = text(str(i))
      bi   = img2se(fi)
      fbi  = dilate(fbin,bi)
      facc = union(facc,fbi)
    show(f,facc)
    darea = blob(fr,'area','data')
    plot([[darea]], [['style','impulses']])'''
    facc=subm(f,f)
    for i in range(1,nblobs+1):
      fbin = cmp(cr,'==',uint16(i))
      fi   = text(str(i))
      bi   = img2se(fi)
      fbi  = dilate(fbin,bi)
      facc = union(facc,fbi)
    show(f,facc)
    darea = blob(fr,'area','data')
    plot([[darea]], [['style','impulses']])
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   brain - Extract the lateral ventricle from an MRI image of the brain.
#
# =========================================================================
def brain():

    print
    print '''Extract the lateral ventricle from an MRI image of the brain.'''
    print
    #
    print '========================================================================='
    print '''
    The MRI image of a brain slice is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('mribrain.tif');
    show(a);'''
    a = readgray('mribrain.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The ventricle is enhanced using an opening with a disk of radius 10
    followed by a reconstruction.
    '''
    print '========================================================================='
    #0
    print '''
    b = open(a,sedisk(10));
    c = infrec(b,a);
    show(b);
    show(c);'''
    b = open(a,sedisk(10));
    c = infrec(b,a);
    show(b);
    show(c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The result of the open by reconstruction is subtracted from the
    original image. Note that the three operations: open, reconstruction
    and the subtraction could be done at once using the (open by
    reconstruction top-hat) function. On the right, the enhanced
    ventricle is thresholded.
    '''
    print '========================================================================='
    #0
    print '''
    d = subm(a,c);
    show(d);
    e = cmp(d,'>=',uint8(50));
    show(e);'''
    d = subm(a,c);
    show(d);
    e = cmp(d,'>=',uint8(50));
    show(e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Finally, the ventricle is selected as the connected object with area
    larger than 70 pixels. For visualization purposes, the result of the
    segmentation is overlayed on the original brain image.
    '''
    print '========================================================================='
    #0
    print '''
    f= areaopen(e,70);
    show(f);
    show(a,f);'''
    f= areaopen(e,70);
    show(f);
    show(a,f);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   calc - Extract the keys of a calculator.
#
# =========================================================================
def calc():

    print
    print '''Extract the keys of a calculator.'''
    print
    #
    print '========================================================================='
    print '''
    The gray-scale image of the calculator is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('keyb.tif');
    show(a);'''
    a = readgray('keyb.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The image edges are enhanced by the gradient operator.
    '''
    print '========================================================================='
    #0
    print '''
    b = gradm(a, sebox());
    show(b);'''
    b = gradm(a, sebox());
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The opening top-hat procedure enhances the small objects relatively
    to its background. In the calculator image, the digits are enhanced.
    '''
    print '========================================================================='
    #0
    print '''
    c = openth(a,sebox(5));
    show(c);'''
    c = openth(a,sebox(5));
    show(c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The threshold operator is used to separated the enhanced objects.
    This procedure is quite robust, since the background was reduced to
    very low levels with the opening top-hat.
    '''
    print '========================================================================='
    #0
    print '''
    d = threshad(c, uint8(150));
    show(d);'''
    d = threshad(c, uint8(150));
    show(d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    In order to have just one object (i.e., connected component) inside
    each key, a dilation is applied.
    '''
    print '========================================================================='
    #0
    print '''
    e = dilate(d, sebox(3));
    show(e);'''
    e = dilate(d, sebox(3));
    show(e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The outside markers are built by taking the watershed (skiz) of the
    complement of internal markers image.
    '''
    print '========================================================================='
    #0
    print '''
    f = watershed(neg(e));
    show(f);'''
    f = watershed(neg(e));
    show(f);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The markers used are the union of the internal and external markers
    detected. They are displayed as overlay on the gradient image.
    '''
    print '========================================================================='
    #0
    print '''
    g = union(e,f);
    show(b,g);'''
    g = union(e,f);
    show(b,g);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The calculator keys are extracted by applying the watershed operator
    on the gradient image, constrained by the markers detected.
    '''
    print '========================================================================='
    #0
    print '''
    h = cwatershed(b,g,sebox());
    show(h);'''
    h = cwatershed(b,g,sebox());
    show(h);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Superposition of the detected contours on the input image.
    '''
    print '========================================================================='
    #0
    print '''
    show(a,h);'''
    show(a,h);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   cells - Extract blood cells and separate them.
#
# =========================================================================
def cells():

    print
    print '''Extract blood cells and separate them.'''
    print
    #
    print '========================================================================='
    print '''
    First, the blood cells image is read. Then, the gray-scale area open
    operator is applied for removing small white pores over the cells.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('bloodcells.tif');
    show(a);
    b = areaopen(a, 200);
    show(b);'''
    a = readgray('bloodcells.tif');
    show(a);
    b = areaopen(a, 200);
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The threshold of dark areas produces the segmented image (i.e., the
    region where there are cells). Then the opening by a small disk
    performs smoothing of the cells borders.
    '''
    print '========================================================================='
    #0
    print '''
    c = cmp( uint8(0), '<=', b, '<=', uint8(140));
    show(c);
    d = open(c,sedisk(2,'2D','OCTAGON'));
    show(d);'''
    c = cmp( uint8(0), '<=', b, '<=', uint8(140));
    show(c);
    d = open(c,sedisk(2,'2D','OCTAGON'));
    show(d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A marker for each cell is obtained by dilating the regional maximum
    of the distance transform. For visualization illustration, the
    distance transform is viewed as a topographic surface shading on the
    left and the dilated regional maximum is displayed in read overlayed
    on the surface view.
    '''
    print '========================================================================='
    #0
    print '''
    e1 = dist(d, sebox(),'EUCLIDEAN');
    e2 = surf(e1);
    show( e2);
    e3 = regmax(e1);
    e  = dilate(e3);
    show( e2, e);'''
    e1 = dist(d, sebox(),'EUCLIDEAN');
    e2 = surf(e1);
    show( e2);
    e3 = regmax(e1);
    e  = dilate(e3);
    show( e2, e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The watershed, constrained by the makers image, is applied to the
    negation of the distance function. The result of this procedure is
    also called geodesic SKIZ. For visualization, on the left the negate
    distance function is displayed as a topographic surface, and on the
    right this surface is superposed by the markers and the detected
    watershed lines.
    '''
    print '========================================================================='
    #0
    print '''
    f = neg(e1);
    fs = surf(f);
    show(fs);
    g = cwatershed( f, e, sebox());
    show(fs, g, e);'''
    f = neg(e1);
    fs = surf(f);
    show(fs);
    g = cwatershed( f, e, sebox());
    show(fs, g, e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The geodesic SKIZ (i.e., watershed division lines) is subtracted
    from the segmented image, separating the cells. On the left the
    detected watershed lines is overlayed on the cells binary image, and
    on the right, the cells image separated using the watershed lines.
    '''
    print '========================================================================='
    #0
    print '''
    show(c,g);
    h = intersec(c,neg(g));
    show(h);'''
    show(c,g);
    h = intersec(c,neg(g));
    show(h);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The cells that touch the frame of the image are removed.
    '''
    print '========================================================================='
    #0
    print '''
    i = edgeoff(h);
    show(i);'''
    i = edgeoff(h);
    show(i);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Superposition of the contour of the detected cells on the original
    image.
    '''
    print '========================================================================='
    #0
    print '''
    j=gradm(i);
    show(a,j);'''
    j=gradm(i);
    show(a,j);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   chickparts - Classify chicken parts in breast, legs+tights and wings
#
# =========================================================================
def chickparts():

    print
    print '''Classify chicken parts in breast, legs+tights and wings'''
    print
    #
    print '========================================================================='
    print '''
    The input image is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('chickparts.tif');
    show(a);'''
    a = readgray('chickparts.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Convert to binary objects by thresholding and then labeling the
    objects.
    '''
    print '========================================================================='
    #0
    print '''
    b = cmp(a,'>=', uint8(100));
    show(b);
    c = label(b);
    lblshow(c,'border');'''
    b = cmp(a,'>=', uint8(100));
    show(b);
    c = label(b);
    lblshow(c,'border');
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Measure the area o each object and put this value as the pixel
    object value. For displaying purpose, overlay the background as red
    in the right image below.
    '''
    print '========================================================================='
    #0
    print '''
    d = blob(c,'area');
    show(d);
    show(d, cmp(d,'==',0));'''
    d = blob(c,'area');
    show(d);
    show(d, cmp(d,'==',0));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The wings are detected by finding objects with area 100 and 2500
    pixels. The tights are selected as connected objects with area
    between 2500 and 5500 pixels.
    '''
    print '========================================================================='
    #0
    print '''
    wings = cmp( uint16(100),'<=',d, '<=', uint16(2500));
    show(wings);
    tights = cmp( uint16(2500),'<',d, '<=', uint16(5500));
    show(tights);'''
    wings = cmp( uint16(100),'<=',d, '<=', uint16(2500));
    show(wings);
    tights = cmp( uint16(2500),'<',d, '<=', uint16(5500));
    show(tights);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The legs+tights have area larger than 5500 and smaller than 8500
    pixels and the breast is the largest connected object with area
    larger than 8500 pixels
    '''
    print '========================================================================='
    #0
    print '''
    legs = cmp( uint16(5500), '<', d, '<=', uint16(8500));
    show(legs);
    breast = cmp( d,'>', uint16(8500));
    show(breast);'''
    legs = cmp( uint16(5500), '<', d, '<=', uint16(8500));
    show(legs);
    breast = cmp( d,'>', uint16(8500));
    show(breast);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Overlay the contour of the detected parts over the original image
    '''
    print '========================================================================='
    #0
    print '''
    show(a, gradm(wings), gradm(tights), gradm(legs),gradm(breast));'''
    show(a, gradm(wings), gradm(tights), gradm(legs),gradm(breast));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   concrete - Aggregate and anhydrous phase extraction from a concrete section observed by a SEM image.
#
# =========================================================================
def concrete():

    print
    print '''Aggregate and anhydrous phase extraction from a concrete section
    observed by a SEM image.'''
    print
    #
    print '========================================================================='
    print '''
    The SEM image of a polished concrete section is read. The anhydrous
    phase are the white pores, while the aggregate are the medium-gray
    homogeneous pores.
    '''
    print '========================================================================='
    #0
    print '''
    f = readgray('csample.jpg')
    show(f)'''
    f = readgray('csample.jpg')
    show(f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The histogram has a small peak in the white region related to the
    anhydrous phase.
    '''
    print '========================================================================='
    #0
    print '''
    h = histogram(f)
    plot([[h]])'''
    h = histogram(f)
    plot([[h]])
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The threshold value is extracted using the watershed technique. The
    aim is to detect the middle valley of the histogram. If the
    histogram is negated, we need to extract the middle peak of the 1D
    signal. This is accomplished by find proper markers on the valleys.
    These markers are extracted by detecting the regional minima of the
    filtered signal (alternating sequential filtering, closing followed
    by opening of length 5 pixels). To discard the detection of peaks
    near the limits of the histogram, an intersection is done using the
    function. For illustrative purpose, a plot of all the signals
    involved is displayed.
    '''
    print '========================================================================='
    #0
    print '''
    hf = asf(neg(h),'co',seline(5))
    ws = watershed(hf)
    wsf = intersec(neg(frame(ws,20)),ws)
    t = nonzero(wsf)
    print t
    ax = stats(h,'max')
    hf_plot = neg(hf)
    ws_plot = gray(ws, 'uint16', ax)
    wsf_plot = gray(wsf, 'uint16', ax)
    plot([[hf_plot],[ws_plot],[wsf_plot]])'''
    hf = asf(neg(h),'co',seline(5))
    ws = watershed(hf)
    wsf = intersec(neg(frame(ws,20)),ws)
    t = nonzero(wsf)
    print t
    ax = stats(h,'max')
    hf_plot = neg(hf)
    ws_plot = gray(ws, 'uint16', ax)
    wsf_plot = gray(wsf, 'uint16', ax)
    plot([[hf_plot],[ws_plot],[wsf_plot]])
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The threshold value found in the previous step is applied. After, a
    filter to remove blobs smaller then 20 pixels is applied. For
    illustrative, the contour of the anhydrous grains are displayed as
    an overlay on the original image.
    '''
    print '========================================================================='
    #0
    print '''
    aux = threshad( f, t, 255)
    anidro = areaopen(aux, 20)
    show( f, gradm(anidro))'''
    aux = threshad( f, t, 255)
    anidro = areaopen(aux, 20)
    show( f, gradm(anidro))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The watershed applied on the gradient using the markers from
    filtered regional minima of the gradient is a standard watershed
    based technique. In this case the filter was chosen to be a contrast
    .
    '''
    print '========================================================================='
    #0
    print '''
    g=gradm(f)
    m=regmin(hmin(g,10))
    ws=cwatershed(g,m)
    show(ws)'''
    g=gradm(f)
    m=regmin(hmin(g,10))
    ws=cwatershed(g,m)
    show(ws)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The result of the watershed in the previous step is the detection of
    a large number of regions. The larger ones are the aggregate and the
    anhydrous. So first the regions are filtered out using an area
    criterion of 300 pixels. Small holes (area <= 50) are closed. The
    aggregate is obtained by removing the anhydrous phase.
    '''
    print '========================================================================='
    #0
    print '''
    aux1=areaopen(neg(ws),300)
    aux2=areaclose(aux1,50)
    aggr=subm(aux2,anidro)
    show(f, gradm(aggr))'''
    aux1=areaopen(neg(ws),300)
    aux2=areaclose(aux1,50)
    aggr=subm(aux2,anidro)
    show(f, gradm(aggr))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Finally each phase is measured and an illustrative display is
    constructed. The grains contoured by red are the aggregate and those
    contoured by green, the anhydrous.
    '''
    print '========================================================================='
    #0
    print '''
    n = product(shape(f))
    anidro_phase = stats(anidro,'sum')/n
    print 'anidro=',anidro_phase
    aggr_phase = stats(aggr,'sum')/n;
    print 'aggr=',aggr_phase
    show( f, gradm(aggr), gradm(anidro))'''
    n = product(shape(f))
    anidro_phase = stats(anidro,'sum')/n
    print 'anidro=',anidro_phase
    aggr_phase = stats(aggr,'sum')/n;
    print 'aggr=',aggr_phase
    show( f, gradm(aggr), gradm(anidro))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   cookies - Detect broken rounded biscuits.
#
# =========================================================================
def cookies():

    print
    print '''Detect broken rounded biscuits.'''
    print
    #
    print '========================================================================='
    print '''
    The input image is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('cookies.tif');
    show(a);'''
    a = readgray('cookies.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Convert to binary objects by thresholding
    '''
    print '========================================================================='
    #0
    print '''
    b = threshad(a, uint8(100));
    show(b);'''
    b = threshad(a, uint8(100));
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The tophat of the binary image by an octagon disk with a radius fits
    the good biscuit but does not fit in the broken biscuit can detect
    the broken one.
    '''
    print '========================================================================='
    #0
    print '''
    c = openth(b,sedisk(55,'2D','OCTAGON'));
    show(c);'''
    c = openth(b,sedisk(55,'2D','OCTAGON'));
    show(c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Clean the residues from the octagon disk and the rounded shaped
    biscuits by eliminating small connected regions
    '''
    print '========================================================================='
    #0
    print '''
    d = areaopen(c,400);
    show(d);'''
    d = areaopen(c,400);
    show(d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Display the detected broken biscuit
    '''
    print '========================================================================='
    #0
    print '''
    show(a,d);'''
    show(a,d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   cornea - Cornea cells marking.
#
# =========================================================================
def cornea():

    print
    print '''Cornea cells marking.'''
    print
    #
    print '========================================================================='
    print '''
    The gray-scale image of the cornea is read and displayed. A
    topographic model is also displayed. We can notice that the cells
    are formed by small hills in the topographic model. We can also
    notice that the image is very noisy.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('corneacells.tif');
    show(a);
    b = surf(a);
    show(b);'''
    a = readgray('corneacells.tif');
    show(a);
    b = surf(a);
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The image is filtered by an alternating sequential filtering with
    size 2. This filter is composed by openings and closings, removing
    small peaks and valleys. Next, the regional maxima are detected. For
    illustrative purpose, they are displayed overlayed on the
    topographic image view. These regional maxima are the markers for
    each cell. If anything goes wrong in this step, the error will be
    propagated throughout the process.
    '''
    print '========================================================================='
    #0
    print '''
    c = asf(a,'oc',secross(),2);
    d = regmax( c);
    show(surf(c));
    show(surf(c), d);'''
    c = asf(a,'oc',secross(),2);
    d = regmax( c);
    show(surf(c));
    show(surf(c), d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Following the paradigm of segmentation by watershed, the background
    marker is detected by applying the constrained watershed on the
    negation of the cells image using the markers detected in the last
    step. These watershed lines partition the image in regions of
    influence of each cell. For illustrative display, the negative of
    the cell image is displayed overlayed by the markers on the left,
    and also overlayed by the watershed lines on the right.
    '''
    print '========================================================================='
    #0
    print '''
    e = neg(a);
    f = cwatershed(e, d, sebox());
    show(e,d);
    show(e,f,d);'''
    e = neg(a);
    f = cwatershed(e, d, sebox());
    show(e,d);
    show(e,f,d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    As the internal and external markers can be touching, we combine the
    external marker with value 1 with the labeling of the internal
    markers added by 1. The labeled marker image is shown on the left.
    The final watershed will be applied on the gradient of the original
    image, which is shown on the right.
    '''
    print '========================================================================='
    #0
    print '''
    g = gray(f, 'uint16', 1);
    h1 = addm(label(d), uint16(1));
    h = intersec(gray(d,'uint16'), h1);
    i = union( g, h);
    lblshow(i);
    j = gradm( a);
    show(j);'''
    g = gray(f, 'uint16', 1);
    h1 = addm(label(d), uint16(1));
    h = intersec(gray(d,'uint16'), h1);
    i = union( g, h);
    lblshow(i);
    j = gradm( a);
    show(j);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Apply the constrained watershed on the gradient from the labeled
    internal and external markers. Show the watershed lines on the left
    and the results overlayed on the original image, on the right.
    '''
    print '========================================================================='
    #0
    print '''
    k = cwatershed(j, i);
    show( k);
    show(a, k, k);'''
    k = cwatershed(j, i);
    show( k);
    show(a, k, k);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   fabric - Detection of vertical weave in fabrics.
#
# =========================================================================
def fabric():

    print
    print '''Detection of vertical weave in fabrics.'''
    print
    #
    print '========================================================================='
    print '''
    The image to be processed is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('fabric.tif');
    show(a);'''
    a = readgray('fabric.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A sequence of dilations (by a disk and two line segments) is applied
    to enhance the white stripes
    '''
    print '========================================================================='
    #0
    print '''
    b = dilate(a,sedisk(4));
    c = dilate(b,seline(25,90));
    d = dilate(c,seline(25,-90));
    show(d);'''
    b = dilate(a,sedisk(4));
    c = dilate(b,seline(25,90));
    d = dilate(c,seline(25,-90));
    show(d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The markers are the regional minima with contrast 25.
    '''
    print '========================================================================='
    #0
    print '''
    e = hmin(d,25);
    f = regmin(e);
    show(f);'''
    e = hmin(d,25);
    f = regmin(e);
    show(f);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Watershed of the original image from the dark stripes markers just
    created. Show the result overlayed on the original image.
    '''
    print '========================================================================='
    #0
    print '''
    g = cwatershed(a,f);
    show(a,dilate(g));'''
    g = cwatershed(a,f);
    show(a,dilate(g));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Create a new marker by taking the union of the dark markers and the
    watershed lines just created. The gradient of the original image is
    computed.
    '''
    print '========================================================================='
    #0
    print '''
    h = union(g,f);
    i = gradm(a);
    show(h);
    show(i);'''
    h = union(g,f);
    i = gradm(a);
    show(h);
    show(i);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The watershed of the gradient of the original image, taking the
    marker just created, gives the extend of the white regions.
    '''
    print '========================================================================='
    #0
    print '''
    j = cwatershed(i,h,sebox());
    show(a,j);'''
    j = cwatershed(i,h,sebox());
    show(a,j);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The watershed regions area computed. The area of each watershed
    regions is measured and displayed.
    '''
    print '========================================================================='
    #0
    print '''
    k = cwatershed(i,h,sebox(),'REGIONS');
    lblshow(k,'border');
    l = blob(k,'area');
    show(l);'''
    k = cwatershed(i,h,sebox(),'REGIONS');
    lblshow(k,'border');
    l = blob(k,'area');
    show(l);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    To select only the fabric spacing, select the regions with small
    area (less than 2000). Label the narrow regions.
    '''
    print '========================================================================='
    #0
    print '''
    m = cmp(l,'<=',2000);
    show(m);
    n = label(m,sebox());
    lblshow(n,'border');'''
    m = cmp(l,'<=',2000);
    show(m);
    n = label(m,sebox());
    lblshow(n,'border');
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Compute the area of each region and plot them. Also display the
    original image for illustration. From the plot, we can notice that
    the two rightmost weave spacing are significantly larger than the
    others.
    '''
    print '========================================================================='
    #0
    print '''
    show(a);
    o = blob(n,'area','data');
    plot([[o]],[['style','impulses']])'''
    show(a);
    o = blob(n,'area','data');
    plot([[o]],[['style','impulses']])
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   fila - Detect Filarial Worms.
#
# =========================================================================
def fila():

    print
    print '''Detect Filarial Worms.'''
    print
    #
    print '========================================================================='
    print '''
    A microscopic gray-scale image, with two filarial worms, is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('danaus.tif');
    show(a);'''
    a = readgray('danaus.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The Close by Reconstruction Top-Hat operator is applied to
    regularize the image background.
    '''
    print '========================================================================='
    #0
    print '''
    b = closerecth(a,sebox(5));
    show(b);'''
    b = closerecth(a,sebox(5));
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The gray-scale opening by the elementary cross is applied to remove
    narrow objects.
    '''
    print '========================================================================='
    #0
    print '''
    c = open(b);
    show(c);'''
    c = open(b);
    show(c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The gray-scale area open operator is applied to remove small
    objects.
    '''
    print '========================================================================='
    #0
    print '''
    d = areaopen(c,200);
    show(d);'''
    d = areaopen(c,200);
    show(d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The threshold operator is applied to extract a reduced set of
    structures that include the two worms present in the image.
    '''
    print '========================================================================='
    #0
    print '''
    e = threshad(d,50);
    show(e);'''
    e = threshad(d,50);
    show(e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The objective of the sequence of transformations, that begin with
    the homotopic skeleton, is to eliminateg the structures that are not
    worms. The information used for the filtering is that the worms are
    longer than any other structure found.
    '''
    print '========================================================================='
    #0
    print '''
    f = thin(e);
    show(f);'''
    f = thin(e);
    show(f);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The first 12 points of the skeleton branches, counting from their
    extremities, are eliminated. The structures that were not eliminated
    will be the markers for extracting the two worms.
    '''
    print '========================================================================='
    #0
    print '''
    g = thin(f,endpoints(), 12);
    show(g);'''
    g = thin(f,endpoints(), 12);
    show(g);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The binary reconstruction operator is applied to reconstruct the
    binary image produced by the threshold from the marker image.
    '''
    print '========================================================================='
    #0
    print '''
    h = infrec(g,e);
    show(h);'''
    h = infrec(g,e);
    show(h);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The structures extracted are overlaid to the input gray-scale image.
    '''
    print '========================================================================='
    #0
    print '''
    show(a,h);'''
    show(a,h);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   flatzone - Flat-zone image simplification by connected filtering.
#
# =========================================================================
def flatzone():

    print
    print '''Flat-zone image simplification by connected filtering.'''
    print
    #
    print '========================================================================='
    print '''
    The input image is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('cameraman.tif')
    show(a)'''
    a = readgray('cameraman.tif')
    show(a)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Obtain the flat zones (8-connectivity) and compute its number. The
    number of flat zones is determined by the maximum labeling value (
    starting from flat zone one).
    '''
    print '========================================================================='
    #0
    print '''
    b = labelflat(a,sebox())
    nfz=stats(b,'max')
    print nfz
    show(a)
    lblshow(b)'''
    b = labelflat(a,sebox())
    nfz=stats(b,'max')
    print nfz
    show(a)
    lblshow(b)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Apply the alternating sequential filter by reconstruction with
    increasing sizes. They constitute a connected pyramid.
    '''
    print '========================================================================='
    #0
    print '''
    c=asfrec(a,'CO',sebox(),sebox(),2)
    d=asfrec(a,'CO',sebox(),sebox(),4)
    e=asfrec(a,'CO',sebox(),sebox(),16)
    show(c)
    show(d)
    show(e)'''
    c=asfrec(a,'CO',sebox(),sebox(),2)
    d=asfrec(a,'CO',sebox(),sebox(),4)
    e=asfrec(a,'CO',sebox(),sebox(),16)
    show(c)
    show(d)
    show(e)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    These figures show the image simplification in a connected pyramid.
    Notice how the shapes are well preserved along the scale space. The
    number of flat zones at each level of the pyramid are computed to
    illustrate the flat zone number reduction.
    '''
    print '========================================================================='
    #0
    print '''
    c_lab=labelflat(c,sebox())
    d_lab=labelflat(d,sebox())
    e_lab=labelflat(e,sebox())
    print stats(c_lab,'max')
    print stats(d_lab,'max')
    print stats(e_lab,'max')
    lblshow(c_lab)
    lblshow(d_lab)
    lblshow(e_lab)'''
    c_lab=labelflat(c,sebox())
    d_lab=labelflat(d,sebox())
    e_lab=labelflat(e,sebox())
    print stats(c_lab,'max')
    print stats(d_lab,'max')
    print stats(e_lab,'max')
    lblshow(c_lab)
    lblshow(d_lab)
    lblshow(e_lab)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    In this experiment we select a particular flat zone, the flat zone
    to which the pixel (90,60) belongs, and display it at each level of
    the connected pyramid. Notice the flat zone inclusion property.
    '''
    print '========================================================================='
    #0
    print '''
    c_v=c_lab[89,59]
    c_flat=cmp(c_lab,'==',c_v)
    d_v=d_lab[89,59]
    d_flat=cmp(d_lab,'==',d_v)
    e_v=e_lab[89,59]
    e_flat=cmp(e_lab,'==',e_v)
    show(a,e_flat,d_flat,c_flat)'''
    c_v=c_lab[89,59]
    c_flat=cmp(c_lab,'==',c_v)
    d_v=d_lab[89,59]
    d_flat=cmp(d_lab,'==',d_v)
    e_v=e_lab[89,59]
    e_flat=cmp(e_lab,'==',e_v)
    show(a,e_flat,d_flat,c_flat)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   flow - Detect water in a static image of an oil-water flow experiment.
#
# =========================================================================
def flow():

    print
    print '''Detect water in a static image of an oil-water flow experiment.'''
    print
    #
    print '========================================================================='
    print '''
    The gray-scale image of the water-oil flow experiment is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('flow.tif')
    show(a)'''
    a = readgray('flow.tif')
    show(a)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The dark region of the image is enhanced by the close top-hat
    operator.
    '''
    print '========================================================================='
    #0
    print '''
    b=closeth(a,seline(50,90))
    show(b)'''
    b=closeth(a,seline(50,90))
    show(b)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A connected filtering is applied to remove small artifacts present
    in the image.
    '''
    print '========================================================================='
    #0
    print '''
    c=closerec(b,sebox(5))
    show(c)'''
    c=closerec(b,sebox(5))
    show(c)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    An alternated sequential filtering is used for shape smoothing.
    '''
    print '========================================================================='
    #0
    print '''
    d=asf(c,'co',secross())
    show(d)'''
    d=asf(c,'co',secross())
    show(d)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The original and thresholded image overlayed on the original are
    presented successively.
    '''
    print '========================================================================='
    #0
    print '''
    e=threshad(d,100)
    show(a)
    show(a,e)'''
    e=threshad(d,100)
    show(a)
    show(a,e)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   gear - Detect the teeth of a gear
#
# =========================================================================
def gear():

    print
    print '''Detect the teeth of a gear'''
    print
    #
    print '========================================================================='
    print '''
    The binary image of a gear is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('gear.tif');
    show(a);'''
    a = readgray('gear.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Opening of the input image by an Euclidean disk of radius 20. The
    sequence opening-subtraction is called opening top-hat. The opening
    top-hat could be executed in a single coand: c = (a,(20));
    '''
    print '========================================================================='
    #0
    print '''
    b = open(a,sedisk(20));
    show(b);
    c = subm(a,b);
    show(c);'''
    b = open(a,sedisk(20));
    show(b);
    c = subm(a,b);
    show(c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The teeth detected are labeled. The maximum pixel value in the
    labeled image gives the number of connected objects (n. of teeth).
    '''
    print '========================================================================='
    #0
    print '''
    d = label(c);
    nteeth=stats(d,'max')
    lblshow(d,'border');'''
    d = label(c);
    nteeth=stats(d,'max')
    lblshow(d,'border');
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   holecenter - Hole center misalignment in PCB.
#
# =========================================================================
def holecenter():

    print
    print '''Hole center misalignment in PCB.'''
    print
    #
    print '========================================================================='
    print '''
    The image of the PCB is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('pcbholes.tif')
    show(a)'''
    a = readgray('pcbholes.tif')
    show(a)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Use the close hole function to remove the holes. Note that one hole
    is open. This is not considered in this experiment. The regional
    maxima of the distance transform gives the radius of the largest
    disk inside the pad. We are interested only in radius larger than 20
    pixels.
    '''
    print '========================================================================='
    #0
    print '''
    b = clohole(a)
    show(b)
    d = dist(b,secross(),'EUCLIDEAN')
    e = regmax(d,sebox())
    f = threshad(d, uint16([20]))   # radius larger than 20 pixels
    g = intersec(e,f)
    h = blob(label(g,sebox()),'CENTROID'); # pad center
    show(b,dilate(h))'''
    b = clohole(a)
    show(b)
    d = dist(b,secross(),'EUCLIDEAN')
    e = regmax(d,sebox())
    f = threshad(d, uint16([20]))   # radius larger than 20 pixels
    g = intersec(e,f)
    h = blob(label(g,sebox()),'CENTROID'); # pad center
    show(b,dilate(h))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The holes are given by the difference of the pad image from the
    original image. Repeat the same procedure to find the center of the
    pads to find now the center of the holes.
    '''
    print '========================================================================='
    #0
    print '''
    i = subm(b,a)
    show(i)
    j = dist(i,secross(),'EUCLIDEAN')
    k = regmax(j,sebox())
    l = blob(label(k,sebox()),'CENTROID') # hole center
    show(i,dilate(l))'''
    i = subm(b,a)
    show(i)
    j = dist(i,secross(),'EUCLIDEAN')
    k = regmax(j,sebox())
    l = blob(label(k,sebox()),'CENTROID') # hole center
    show(i,dilate(l))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    First both centers (pads and holes) are displayed together. Then the
    actual misalignment is computed using the distance from one point to
    the other.
    '''
    print '========================================================================='
    #0
    print '''
    from numpy.oldnumeric import nonzero
    show(a,h,l)
    m = dist(neg(l),secross(),'EUCLIDEAN')
    n = intersec(gray(h),uint8(m))
    show(n,a)
    i = nonzero(n.ravel())
    x = i / n.shape[1]
    y = i % n.shape[1]
    for k in range(len(i)):
      print 'displacement of %d at (%d,%d)\n' %(n[x[k],y[k]],x[k],y[k])'''
    from numpy.oldnumeric import nonzero
    show(a,h,l)
    m = dist(neg(l),secross(),'EUCLIDEAN')
    n = intersec(gray(h),uint8(m))
    show(n,a)
    i = nonzero(n.ravel())
    x = i / n.shape[1]
    y = i % n.shape[1]
    for k in range(len(i)):
      print 'displacement of %d at (%d,%d)\n' %(n[x[k],y[k]],x[k],y[k])
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    First, the thinning to compute the skeleton of the PCB image, then
    remove iteratively all the end points of the skeleton so just the
    skeleton loop around the holes remains. Find the minimum distance of
    these loops to the border and display their location.
    '''
    print '========================================================================='
    #0
    print '''
    o=thin(a)
    p=thin(o,endpoints())
    show(a,p)
    q = dist(a,secross(),'EUCLIDEAN')
    r = grain(label(p,sebox()),q,'min') # minimum
    s = grain(label(p,sebox()),q,'min','data') # minimum
    from numpy.oldnumeric import ravel
    for k in ravel(s):
      print 'Minimum distance: %d pixels' %(2*k+1)
    t = intersec(cmp(r,'==',q),a)
    show(a,dilate(t))'''
    o=thin(a)
    p=thin(o,endpoints())
    show(a,p)
    q = dist(a,secross(),'EUCLIDEAN')
    r = grain(label(p,sebox()),q,'min') # minimum
    s = grain(label(p,sebox()),q,'min','data') # minimum
    from numpy.oldnumeric import ravel
    for k in ravel(s):
      print 'Minimum distance: %d pixels' %(2*k+1)
    t = intersec(cmp(r,'==',q),a)
    show(a,dilate(t))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   leaf - Segment a leaf from the background
#
# =========================================================================
def leaf():

    print
    print '''Segment a leaf from the background'''
    print
    #
    print '========================================================================='
    print '''
    The gray scale image to be processed is read.
    '''
    print '========================================================================='
    #0
    print '''
    f = readgray('leaf.tif')
    show(f)'''
    f = readgray('leaf.tif')
    show(f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Although the leaf was pictured on a light background, it is not
    possible to fully segment the leaf using a simple thresholding
    '''
    print '========================================================================='
    #0
    print '''
    f_low=threshad(f,100)
    f_med=threshad(f,128)
    f_high=threshad(f,160)
    show(f_low)
    show(f_med)
    show(f_high)'''
    f_low=threshad(f,100)
    f_med=threshad(f,128)
    f_high=threshad(f,160)
    show(f_low)
    show(f_med)
    show(f_high)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The toggle is a non-linear image enhancement that changes the pixel
    value to the maximum or the minimum in the neighborhood given by the
    structure element, depending which one is the nearest value. The
    result of the toggle is that near the edges, the image is better
    defined.
    '''
    print '========================================================================='
    #0
    print '''
    f1=toggle(f,erode(f,sedisk(7)),dilate(f,sedisk(7)))
    show(f1)'''
    f1=toggle(f,erode(f,sedisk(7)),dilate(f,sedisk(7)))
    show(f1)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The thresholding can now be applied resulting a good definition of
    the leaf boarder. Small white spots can be removed by the area open
    filter.
    '''
    print '========================================================================='
    #0
    print '''
    f2=threshad(f1,100)
    f3=areaopen(f2,80)
    show(f2)
    show(f3)'''
    f2=threshad(f1,100)
    f3=areaopen(f2,80)
    show(f2)
    show(f3)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    For illustration, the contour of the segmented leaf is overlayed in
    red in the original image
    '''
    print '========================================================================='
    #0
    print '''
    f4=gradm(f3)
    show(f,f4)'''
    f4=gradm(f3)
    show(f,f4)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   lith - Detect defects in a microelectronic circuit.
#
# =========================================================================
def lith():

    print
    print '''Detect defects in a microelectronic circuit.'''
    print
    #
    print '========================================================================='
    print '''
    The input image is read. The image is also displayed as a surface
    model.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('r4x2_256.tif');
    show(a);
    show(surf(a));'''
    a = readgray('r4x2_256.tif');
    show(a);
    show(surf(a));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Closing of the image by a vertical line of length 25 pixels. Then
    subtract it from the original. The sequence closing-subtraction is
    called closing top-hat. (This could be executed in a single coand:
    c=(a,(25,90));).
    '''
    print '========================================================================='
    #0
    print '''
    b = close(a,seline(25,90));
    show(b);
    show(surf(b));'''
    b = close(a,seline(25,90));
    show(b);
    show(surf(b));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Subtraction of the closing from the original is called closing
    top-hat. It shows the discrepancies of the image where the
    structuring element cannot fit the surface. In this case, it
    highlights vertical depression with length longer than 25 pixels.
    '''
    print '========================================================================='
    #0
    print '''
    c = subm(b,a);
    show(c);
    show(surf(c));'''
    c = subm(b,a);
    show(c);
    show(surf(c));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Threshold on the residues image. Elimination of the small objects by
    area open.
    '''
    print '========================================================================='
    #0
    print '''
    d = cmp(c,'>=',50);
    e = areaopen(d,5);
    show(d);
    show(e);'''
    d = cmp(c,'>=',50);
    e = areaopen(d,5);
    show(d);
    show(e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Overlay the detected defects over the original image, and over the
    surface display.
    '''
    print '========================================================================='
    #0
    print '''
    show(a,e);
    show(surf(a),e);'''
    show(a,e);
    show(surf(a),e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   pcb - Decompose a printed circuit board in its main parts.
#
# =========================================================================
def pcb():

    print
    print '''Decompose a printed circuit board in its main parts.'''
    print
    #
    print '========================================================================='
    print '''
    The binary image of a printed circuit board is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('pcb1bin.tif');
    show(a);'''
    a = readgray('pcb1bin.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A new image is created by filling the holes. The input image is
    subtracted from this new image with holes. The resulting residues
    are the holes.
    '''
    print '========================================================================='
    #0
    print '''
    b = clohole(a);
    holes = subm(b,a);
    show(b);
    show(a, holes);'''
    b = clohole(a);
    holes = subm(b,a);
    show(b);
    show(a, holes);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The square islands are detected using an opening by a square of size
    17x17.
    '''
    print '========================================================================='
    #0
    print '''
    c = open(b,sebox(8));
    square = cdil(c, a);
    show(b, c);
    show(holes, square);'''
    c = open(b,sebox(8));
    square = cdil(c, a);
    show(b, c);
    show(holes, square);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The circle islands are detected using an opening by an Euclidean
    disk on a residues image.
    '''
    print '========================================================================='
    #0
    print '''
    f = subm(b, c);
    g = open(f, sedisk(8));
    circle = cdil(g,a);
    show(f, g);
    show(holes, square, circle);'''
    f = subm(b, c);
    g = open(f, sedisk(8));
    circle = cdil(g,a);
    show(f, g);
    show(holes, square, circle);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The rectangular islands are detected using an opening by a rectangle
    of size 25 x 8 on a residues image. The rectangle structuring
    element is built from the composition of vertical and horizontal
    lines.
    '''
    print '========================================================================='
    #0
    print '''
    i = subm(f, g);
    m = open(i,sedil(seline(8,90), seline(25)));
    rect = cdil(m,a);
    show(i, m);
    show(holes, square, circle, rect);'''
    i = subm(f, g);
    m = open(i,sedil(seline(8,90), seline(25)));
    rect = cdil(m,a);
    show(i, m);
    show(holes, square, circle, rect);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The thick connections are detected using an opening by a square on a
    residues image.
    '''
    print '========================================================================='
    #0
    print '''
    o = subm(i,m);
    p = open(o, sebox(2));
    thin = cdil(p,a);
    show(o, p);
    show(holes, square, circle, rect, thin);'''
    o = subm(i,m);
    p = open(o, sebox(2));
    thin = cdil(p,a);
    show(o, p);
    show(holes, square, circle, rect, thin);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The thin connections are detected using an opening by a square on a
    residues image.
    '''
    print '========================================================================='
    #0
    print '''
    r = subm(o,p);
    s = open(r, sebox());
    thick = cdil(s,a);
    show(r, s);
    show(holes, square, circle, rect, thin, thick);'''
    r = subm(o,p);
    s = open(r, sebox());
    thick = cdil(s,a);
    show(r, s);
    show(holes, square, circle, rect, thin, thick);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The main components of the circuit are overlayed and presented in a
    single image.
    '''
    print '========================================================================='
    #0
    print '''
    show(holes, square, circle, rect, thin, thick);'''
    show(holes, square, circle, rect, thin, thick);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   pieces - Classify two dimensional pieces.
#
# =========================================================================
def pieces():

    print
    print '''Classify two dimensional pieces.'''
    print
    #
    print '========================================================================='
    print '''
    The binary image of the pieces is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('pieces_bw.tif');
    show(a);'''
    a = readgray('pieces_bw.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    An homotopic thickening is applied to reduce contour noise.
    '''
    print '========================================================================='
    #0
    print '''
    seA = img2se(binary([[0,1,0],[1,0,1],[0,0,0]]))
    seB = img2se(binary([[0,0,0],[0,1,0],[0,0,0]]))
    iAB = se2hmt(seA,seB);
    print intershow(iAB)
    b = thick(a, iAB);
    show(b);'''
    seA = img2se(binary([[0,1,0],[1,0,1],[0,0,0]]))
    seB = img2se(binary([[0,0,0],[0,1,0],[0,0,0]]))
    iAB = se2hmt(seA,seB);
    print intershow(iAB)
    b = thick(a, iAB);
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The homotopic skeleton by thinning is created.
    '''
    print '========================================================================='
    #0
    print '''
    c = thin(b);
    show(c);'''
    c = thin(b);
    show(c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The open lines of the skeleton are pruned by the end point thinning.
    The remaining skeleton components will be loops, identifying the
    rings.
    '''
    print '========================================================================='
    #0
    print '''
    d = thin(c,endpoints());
    show(c,d);'''
    d = thin(c,endpoints());
    show(c,d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Extraction of the rings by reconstruction of the thicked image from
    the filtered skeleton.
    '''
    print '========================================================================='
    #0
    print '''
    e = infrec(d,b);
    show(e);'''
    e = infrec(d,b);
    show(e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Restriction of the objects detected to the input-image.
    '''
    print '========================================================================='
    #0
    print '''
    f = intersec(a,e);
    show(f);'''
    f = intersec(a,e);
    show(f);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It eliminates the skeleton of the rings.
    '''
    print '========================================================================='
    #0
    print '''
    g = subm(c,e);
    show(g);'''
    g = subm(c,e);
    show(g);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It removes sucessively 4 end-points to let T junctions just on
    T-pins.
    '''
    print '========================================================================='
    #0
    print '''
    h = thin(g, endpoints(), 4);
    show(h);'''
    h = thin(g, endpoints(), 4);
    show(h);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It detects triple points, applying the union of matchings with two
    templates. These points will identify (mark) the T-pins.
    '''
    print '========================================================================='
    #0
    print '''
    seA1 = img2se(binary([[0,1,0],[0,1,0],[1,0,1]]))
    seB1 = img2se(binary([[0,0,0],[1,0,1],[0,1,0]]))
    seA2 = img2se(binary([[0,1,0],[1,1,1],[0,0,0]]))
    seB2 = img2se(binary([[1,0,1],[0,0,0],[0,1,0]]))
    i1 = supcanon(h, se2hmt(seA1,seB1));
    i2 = supcanon(h, se2hmt(seA2,seB2));
    i = union(i1,i2);
    show(h,dilate(i,sedisk(2)));'''
    seA1 = img2se(binary([[0,1,0],[0,1,0],[1,0,1]]))
    seB1 = img2se(binary([[0,0,0],[1,0,1],[0,1,0]]))
    seA2 = img2se(binary([[0,1,0],[1,1,1],[0,0,0]]))
    seB2 = img2se(binary([[1,0,1],[0,0,0],[0,1,0]]))
    i1 = supcanon(h, se2hmt(seA1,seB1));
    i2 = supcanon(h, se2hmt(seA2,seB2));
    i = union(i1,i2);
    show(h,dilate(i,sedisk(2)));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Detection of the T-pins by reconstruction of the ticked image from
    the T-pin markers.
    '''
    print '========================================================================='
    #0
    print '''
    j = infrec(i,b,sebox());
    show(j);'''
    j = infrec(i,b,sebox());
    show(j);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Restriction of the objects detect to the input image
    '''
    print '========================================================================='
    #0
    print '''
    k = intersec(a,j);
    show(k);'''
    k = intersec(a,j);
    show(k);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The nails are imediatly detected by the subtration of the images of
    the rings and T-pints from the input image.
    '''
    print '========================================================================='
    #0
    print '''
    l = subm(subm(a,f),k);
    show(l);'''
    l = subm(subm(a,f),k);
    show(l);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The result of the classification is presented in a pseudo color
    image.
    '''
    print '========================================================================='
    #0
    print '''
    m = gray(f,'uint8',1);
    n = gray(k,'uint8',2);
    o = gray(l,'uint8',3);
    p = union(m,n,o);
    lblshow(p);'''
    m = gray(f,'uint8',1);
    n = gray(k,'uint8',2);
    o = gray(l,'uint8',3);
    p = union(m,n,o);
    lblshow(p);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   potatoes - Grade potato quality by shape and skin spots.
#
# =========================================================================
def potatoes():

    print
    print '''Grade potato quality by shape and skin spots.'''
    print
    #
    print '========================================================================='
    print '''
    The input image is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('potatoes.tif');
    show(a);'''
    a = readgray('potatoes.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Convert to binary objects by thresholding
    '''
    print '========================================================================='
    #0
    print '''
    b = threshad(a,90);
    show(b);'''
    b = threshad(a,90);
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The binary image is thinned and the result overlayed on the original
    image
    '''
    print '========================================================================='
    #0
    print '''
    c = thin(b);
    show(a,c);'''
    c = thin(b);
    show(a,c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    To detect the skin spots, a closing tophat can enhance the dark
    areas of the image
    '''
    print '========================================================================='
    #0
    print '''
    d = closeth(a,sedisk(5));
    show(d);'''
    d = closeth(a,sedisk(5));
    show(d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The tophat is thresholded and the result is masked with the binary
    image of the potatoes as we are interested only on the spots inside
    them
    '''
    print '========================================================================='
    #0
    print '''
    e = threshad(d,20);
    f = intersec(e,b);
    show(f);'''
    e = threshad(d,20);
    f = intersec(e,b);
    show(f);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Show both results: skeleton and skin spots overlayed on the original
    image
    '''
    print '========================================================================='
    #0
    print '''
    show(a);
    show(a,f,c);'''
    show(a);
    show(a,f,c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   robotop - Detect marks on a robot.
#
# =========================================================================
def robotop():

    print
    print '''Detect marks on a robot.'''
    print
    #
    print '========================================================================='
    print '''
    The gray-scale image of the robot top view is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('robotop.tif');
    show(a);'''
    a = readgray('robotop.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It detects white regions smaller than a square of radius 4.
    '''
    print '========================================================================='
    #0
    print '''
    b = openth(a,sebox(4));
    show(b);'''
    b = openth(a,sebox(4));
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It removes white objects smaller than a square of radius 1.
    '''
    print '========================================================================='
    #0
    print '''
    c = open(b,sebox());
    show(c);'''
    c = open(b,sebox());
    show(c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It detects the robot markers. This is a very robust thresholding
    (i.e., the result is not sensible to small changes in the value of
    the threshold parameter). The original image is overlayed by the
    detected robot markers.
    '''
    print '========================================================================='
    #0
    print '''
    d = threshad(c,100);
    show(a,d);'''
    d = threshad(c,100);
    show(a,d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   ruler - Detect defects in a ruler.
#
# =========================================================================
def ruler():

    print
    print '''Detect defects in a ruler.'''
    print
    #
    print '========================================================================='
    print '''
    The gray-scale image of the ruler is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('3.tif');
    show(a);'''
    a = readgray('3.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The close top-hat operator followed by a thresholding is applied.
    '''
    print '========================================================================='
    #0
    print '''
    b = threshad( closeth(a,sebox(5)),40);
    show(b);'''
    b = threshad( closeth(a,sebox(5)),40);
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The vertical lines longer than 50 pixels are detected.
    '''
    print '========================================================================='
    #0
    print '''
    c = open(b,seline(50,90));
    show(c);'''
    c = open(b,seline(50,90));
    show(c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It closes ruler tick marks gaps.
    '''
    print '========================================================================='
    #0
    print '''
    d =close(c,seline(15));
    show(d);'''
    d =close(c,seline(15));
    show(d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It detects all objects connected to the ruler tick markers.
    '''
    print '========================================================================='
    #0
    print '''
    e = infrec(d,b);
    show(e);'''
    e = infrec(d,b);
    show(e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It detects all objects vertically connected to the ruler tick mark.
    Note that the 3x1 rectangle is used as structuring element in the
    vertical reconstruction.
    '''
    print '========================================================================='
    #0
    print '''
    f = infrec(d,b,seline(3,90));
    show(f);'''
    f = infrec(d,b,seline(3,90));
    show(f);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The residues obtained from the previous image.
    '''
    print '========================================================================='
    #0
    print '''
    g = subm(e,f);
    show(g);'''
    g = subm(e,f);
    show(g);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It uses an opening by an elementary cross structuring element to
    eliminate the artifacts.
    '''
    print '========================================================================='
    #0
    print '''
    h = open(g);
    show(h);'''
    h = open(g);
    show(h);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It detects the objects connected to ruler tick marks. A
    reconstruction from the ruler marks detected is applied.
    '''
    print '========================================================================='
    #0
    print '''
    i = infrec(h, b);
    show(i);'''
    i = infrec(h, b);
    show(i);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Overlay the detected defect over the original image
    '''
    print '========================================================================='
    #0
    print '''
    show(a,i);'''
    show(a,i);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   soil - Detect fractures in soil.
#
# =========================================================================
def soil():

    print
    print '''Detect fractures in soil.'''
    print
    #
    print '========================================================================='
    print '''
    The image of fractures in soil is read.
    '''
    print '========================================================================='
    #0
    print '''
    a = readgray('soil.tif');
    show(a);'''
    a = readgray('soil.tif');
    show(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The fracture lines are enhanced by the close top-hat operator.
    '''
    print '========================================================================='
    #0
    print '''
    b = closeth(a,sebox(2));
    show(b);'''
    b = closeth(a,sebox(2));
    show(b);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Small connected bright regions are removed by the gray-scale area
    open operator. Note the connectivity used (: 8-connected).
    '''
    print '========================================================================='
    #0
    print '''
    c= areaopen(b,80,sebox());
    show(c);'''
    c= areaopen(b,80,sebox());
    show(c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The fracture lines are detected. This threshold is very robust.
    '''
    print '========================================================================='
    #0
    print '''
    d = threshad(c,15);
    show(d);'''
    d = threshad(c,15);
    show(d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Overlay of the fracture lines over the original image.
    '''
    print '========================================================================='
    #0
    print '''
    show(a,d);'''
    show(a,d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return

# =====================================================================
#
#   script execution
#
# =====================================================================

#all demonstrations - initialization
_alldemos = [
    'airport',
    'area',
    'asp',
    'labeltext',
    'beef',
    'blob',
    'brain',
    'calc',
    'cells',
    'chickparts',
    'concrete',
    'cookies',
    'cornea',
    'fabric',
    'fila',
    'flatzone',
    'flow',
    'gear',
    'holecenter',
    'leaf',
    'lith',
    'pcb',
    'pieces',
    'potatoes',
    'robotop',
    'ruler',
    'soil',
    ]

def main():
    import sys
    print '\npymorph Demonstrations -- SDC Morphology Toolbox\n'
    print 'Available Demonstrations: \n' + str(_alldemos) + '\n'
    if len(sys.argv) > 1:
        for demo in sys.argv[1:]:
            if demo in _alldemos:
                eval(demo + '()')
            else:
                print "Demonstration " + demo + " is not in this package. Please use help for details\n"
    else:
        print "\nUsage: python %s <demo_name>\n\n" % sys.argv[0]

if __name__ == '__main__':
    main()
