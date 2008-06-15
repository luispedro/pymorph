"""
    Module morphdemo -- Demonstrations
    -------------------------------------------------------------------
    morphdemo is a set of Demonstrations for morph package
    SDC Morphology Toolbox
    -------------------------------------------------------------------
    mmdairport()     -- Detecting runways in satellite airport imagery.
    mmdarea()        -- Remove objects with small areas in binary images.
    mmdasp()         -- Detect the missing aspirin tablets in a card of aspirin
                        tablets.
    mmdbeef()        -- Detect the lean meat region in a beef steak image.
    mmdblob()        -- Demonstrate blob measurements and display.
    mmdbrain()       -- Extract the lateral ventricle from an MRI image of the
                        brain.
    mmdcalc()        -- Extract the keys of a calculator.
    mmdcells()       -- Extract blood cells and separate them.
    mmdchickparts()  -- Classify chicken parts in breast, legs+tights and wings
    mmdconcrete()    -- Aggregate and anhydrous phase extraction from a concrete
                        section observed by a SEM image.
    mmdcookies()     -- Detect broken rounded biscuits.
    mmdcornea()      -- Cornea cells marking.
    mmdfabric()      -- Detection of vertical weave in fabrics.
    mmdfila()        -- Detect Filarial Worms.
    mmdflatzone()    -- Flat-zone image simplification by connected filtering.
    mmdflow()        -- Detect water in a static image of an oil-water flow
                        experiment.
    mmdgear()        -- Detect the teeth of a gear
    mmdholecenter()  -- Hole center misalignment in PCB.
    mmdlabeltext()   -- Segmenting letters, words and paragraphs.
    mmdleaf()        -- Segment a leaf from the background
    mmdlith()        -- Detect defects in a microelectronic circuit.
    mmdpcb()         -- Decompose a printed circuit board in its main parts.
    mmdpieces()      -- Classify two dimensional pieces.
    mmdpotatoes()    -- Grade potato quality by shape and skin spots.
    mmdrobotop()     -- Detect marks on a robot.
    mmdruler()       -- Detect defects in a ruler.
    mmdsoil()        -- Detect fractures in soil.
"""
from pymorph.compat import *
import numpy
# =========================================================================
#
#   mmdairport - Detecting runways in satellite airport imagery.
#
# =========================================================================
def mmdairport():

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
    f = mmreadgray('galeao.jpg')
    mmshow(f)'''
    f = mmreadgray('galeao.jpg')
    mmshow(f)
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
    th=mmopenth(f,mmsedisk(5))
    mmshow(mmaddm(th, 150))'''
    th=mmopenth(f,mmsedisk(5))
    mmshow(mmaddm(th, 150))
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
    bin=mmthreshad(th,30)
    mmshow(f,bin)'''
    bin=mmthreshad(th,30)
    mmshow(f,bin)
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
    m1=mmthin(bin)
    m2=mmthin(m1,mmendpoints())
    m=mmareaopen(m2,1000,mmsebox())
    mmshow(f,m1,m2,m)'''
    m1=mmthin(bin)
    m2=mmthin(m1,mmendpoints())
    m=mmareaopen(m2,1000,mmsebox())
    mmshow(f,m1,m2,m)
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
    g=mminfrec(mmgray(m), th)
    mmshow(g)'''
    g=mminfrec(mmgray(m), th)
    mmshow(g)
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
    final=mmthreshad(g, 20)
    mmshow(f, final)'''
    final=mmthreshad(g, 20)
    mmshow(f, final)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdarea - Remove objects with small areas in binary images.
#
# =========================================================================
def mmdarea():

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
    a = mmreadgray('circuit_bw.tif')
    mmshow(a)'''
    a = mmreadgray('circuit_bw.tif')
    mmshow(a)
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
    b = mmareaopen(a,200)
    mmshow(b)'''
    b = mmareaopen(a,200)
    mmshow(b)
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
    mmshow(a,b)'''
    mmshow(a,b)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdasp - Detect the missing aspirin tablets in a card of aspirin tablets.
#
# =========================================================================
def mmdasp():

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
    a = mmreadgray('astablet.tif')
    mmshow(a)'''
    a = mmreadgray('astablet.tif')
    mmshow(a)
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
    b = mmsurf(a)
    mmshow(b)
    c = mmregmax(a,mmsebox())
    mmshow(b,c)'''
    b = mmsurf(a)
    mmshow(b)
    c = mmregmax(a,mmsebox())
    mmshow(b,c)
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
    d = mmopen(a, mmsedisk(20))
    e = mmsurf(d)
    mmshow(e)
    f = mmregmax(d,mmsebox())
    mmshow(e,f)'''
    d = mmopen(a, mmsedisk(20))
    e = mmsurf(d)
    mmshow(e)
    f = mmregmax(d,mmsebox())
    mmshow(e,f)
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
    mmshow(a)
    mmshow(f)'''
    mmshow(a)
    mmshow(f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdlabeltext - Segmenting letters, words and paragraphs.
#
# =========================================================================
def mmdlabeltext():

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
    f = mmreadgray('stext.tif')
    mmshow(f)'''
    f = mmreadgray('stext.tif')
    mmshow(f)
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
    fl=mmlabel(f,mmsebox())
    mmlblshow(fl)'''
    fl=mmlabel(f,mmsebox())
    mmlblshow(fl)
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
    sew = mmimg2se(mmbinary(ones((7,11))))
    mmseshow(sew)
    fw=mmlabel(f,sew)
    mmlblshow(fw)'''
    from numpy.oldnumeric import ones
    sew = mmimg2se(mmbinary(ones((7,11))))
    mmseshow(sew)
    fw=mmlabel(f,sew)
    mmlblshow(fw)
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
    sep = mmimg2se(mmbinary(ones((20,35))))
    fp=mmlabel(f,sep)
    mmlblshow(fp)'''
    sep = mmimg2se(mmbinary(ones((20,35))))
    fp=mmlabel(f,sep)
    mmlblshow(fp)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdbeef - Detect the lean meat region in a beef steak image.
#
# =========================================================================
def mmdbeef():

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
    a = mmreadgray('beef.tif');
    mmshow(a);'''
    a = mmreadgray('beef.tif');
    mmshow(a);
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
    b=mmclose(a,mmsedisk(2));
    mmshow(b);'''
    b=mmclose(a,mmsedisk(2));
    mmshow(b);
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
    c = mmthreshad(a,uint8(10));
    d = mmareaclose(c,200);
    mmshow(d);'''
    c = mmthreshad(a,uint8(10));
    d = mmareaclose(c,200);
    mmshow(d);
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
    e = mmgradm(d,mmsecross(1),mmsebox(13));
    mmshow(e);'''
    e = mmgradm(d,mmsecross(1),mmsebox(13));
    mmshow(e);
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
    f= mmero(d,mmsecross(80));
    g = mmunion(e,f);  
    h = mmgradm(b);
    mmshow(h,g);'''
    f= mmero(d,mmsecross(80));
    g = mmunion(e,f);  
    h = mmgradm(b);
    mmshow(h,g);
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
    i=mmcwatershed(h,g); 
    mmshow(i);'''
    i=mmcwatershed(h,g); 
    mmshow(i);
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
    mmshow(a,mmdil(i));'''
    mmshow(a,mmdil(i));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdblob - Demonstrate blob measurements and display.
#
# =========================================================================
def mmdblob():

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
    f  = mmreadgray('blob3.tif')
    fr = mmlabel(f)
    mmshow(f)
    mmlblshow(fr,'border')
    nblobs=mmstats(fr,'max')
    print nblobs'''
    f  = mmreadgray('blob3.tif')
    fr = mmlabel(f)
    mmshow(f)
    mmlblshow(fr,'border')
    nblobs=mmstats(fr,'max')
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
    c  = mmblob(fr,'centroid')
    cr = mmlabel(c)
    mmshow(f,c)
    mmlblshow(mmdil(cr))'''
    c  = mmblob(fr,'centroid')
    cr = mmlabel(c)
    mmshow(f,c)
    mmlblshow(mmdil(cr))
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
    fbin = mmcmp(cr,'==',uint16(5))
    f5   = mmtext('5')
    print f5
    b5   = mmimg2se(f5)
    fb5  = mmdil(fbin,b5)
    mmshow(mmdil(fbin))
    mmshow(f,fb5)'''
    fbin = mmcmp(cr,'==',uint16(5))
    f5   = mmtext('5')
    print f5
    b5   = mmimg2se(f5)
    fb5  = mmdil(fbin,b5)
    mmshow(mmdil(fbin))
    mmshow(f,fb5)
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
    facc=mmsubm(f,f)
    for i in range(1,nblobs+1):
      fbin = mmcmp(cr,'==',uint16(i))
      fi   = mmtext(str(i))
      bi   = mmimg2se(fi)
      fbi  = mmdil(fbin,bi)
      facc = mmunion(facc,fbi)
    mmshow(f,facc)
    darea = mmblob(fr,'area','data')
    mmplot([[darea]], [['style','impulses']])'''
    facc=mmsubm(f,f)
    for i in range(1,nblobs+1):
      fbin = mmcmp(cr,'==',uint16(i))
      fi   = mmtext(str(i))
      bi   = mmimg2se(fi)
      fbi  = mmdil(fbin,bi)
      facc = mmunion(facc,fbi)
    mmshow(f,facc)
    darea = mmblob(fr,'area','data')
    mmplot([[darea]], [['style','impulses']])
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdbrain - Extract the lateral ventricle from an MRI image of the brain.
#
# =========================================================================
def mmdbrain():

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
    a = mmreadgray('mribrain.tif');
    mmshow(a);'''
    a = mmreadgray('mribrain.tif');
    mmshow(a);
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
    b = mmopen(a,mmsedisk(10));
    c = mminfrec(b,a);
    mmshow(b);
    mmshow(c);'''
    b = mmopen(a,mmsedisk(10));
    c = mminfrec(b,a);
    mmshow(b);
    mmshow(c);
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
    d = mmsubm(a,c);
    mmshow(d);
    e = mmcmp(d,'>=',uint8(50));
    mmshow(e);'''
    d = mmsubm(a,c);
    mmshow(d);
    e = mmcmp(d,'>=',uint8(50));
    mmshow(e);
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
    f= mmareaopen(e,70);
    mmshow(f);
    mmshow(a,f);'''
    f= mmareaopen(e,70);
    mmshow(f);
    mmshow(a,f);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdcalc - Extract the keys of a calculator.
#
# =========================================================================
def mmdcalc():

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
    a = mmreadgray('keyb.tif');
    mmshow(a);'''
    a = mmreadgray('keyb.tif');
    mmshow(a);
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
    b = mmgradm(a, mmsebox());
    mmshow(b);'''
    b = mmgradm(a, mmsebox());
    mmshow(b);
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
    c = mmopenth(a,mmsebox(5));
    mmshow(c);'''
    c = mmopenth(a,mmsebox(5));
    mmshow(c);
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
    d = mmthreshad(c, uint8(150));
    mmshow(d);'''
    d = mmthreshad(c, uint8(150));
    mmshow(d);
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
    e = mmdil(d, mmsebox(3));
    mmshow(e);'''
    e = mmdil(d, mmsebox(3));
    mmshow(e);
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
    f = mmwatershed(mmneg(e));
    mmshow(f);'''
    f = mmwatershed(mmneg(e));
    mmshow(f);
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
    g = mmunion(e,f);
    mmshow(b,g);'''
    g = mmunion(e,f);
    mmshow(b,g);
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
    h = mmcwatershed(b,g,mmsebox());
    mmshow(h);'''
    h = mmcwatershed(b,g,mmsebox());
    mmshow(h);
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
    mmshow(a,h);'''
    mmshow(a,h);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdcells - Extract blood cells and separate them.
#
# =========================================================================
def mmdcells():

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
    a = mmreadgray('bloodcells.tif');
    mmshow(a);
    b = mmareaopen(a, 200);
    mmshow(b);'''
    a = mmreadgray('bloodcells.tif');
    mmshow(a);
    b = mmareaopen(a, 200);
    mmshow(b);
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
    c = mmcmp( uint8(0), '<=', b, '<=', uint8(140));
    mmshow(c);
    d = mmopen(c,mmsedisk(2,'2D','OCTAGON'));
    mmshow(d);'''
    c = mmcmp( uint8(0), '<=', b, '<=', uint8(140));
    mmshow(c);
    d = mmopen(c,mmsedisk(2,'2D','OCTAGON'));
    mmshow(d);
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
    e1 = mmdist(d, mmsebox(),'EUCLIDEAN');
    e2 = mmsurf(e1);
    mmshow( e2);
    e3 = mmregmax(e1);
    e  = mmdil(e3);
    mmshow( e2, e);'''
    e1 = mmdist(d, mmsebox(),'EUCLIDEAN');
    e2 = mmsurf(e1);
    mmshow( e2);
    e3 = mmregmax(e1);
    e  = mmdil(e3);
    mmshow( e2, e);
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
    f = mmneg(e1);
    fs = mmsurf(f);
    mmshow(fs);
    g = mmcwatershed( f, e, mmsebox());
    mmshow(fs, g, e);'''
    f = mmneg(e1);
    fs = mmsurf(f);
    mmshow(fs);
    g = mmcwatershed( f, e, mmsebox());
    mmshow(fs, g, e);
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
    mmshow(c,g);
    h = mmintersec(c,mmneg(g));
    mmshow(h);'''
    mmshow(c,g);
    h = mmintersec(c,mmneg(g));
    mmshow(h);
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
    i = mmedgeoff(h);
    mmshow(i);'''
    i = mmedgeoff(h);
    mmshow(i);
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
    j=mmgradm(i);
    mmshow(a,j);'''
    j=mmgradm(i);
    mmshow(a,j);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdchickparts - Classify chicken parts in breast, legs+tights and wings
#
# =========================================================================
def mmdchickparts():

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
    a = mmreadgray('chickparts.tif');
    mmshow(a);'''
    a = mmreadgray('chickparts.tif');
    mmshow(a);
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
    b = mmcmp(a,'>=', uint8(100));
    mmshow(b);
    c = mmlabel(b);
    mmlblshow(c,'border');'''
    b = mmcmp(a,'>=', uint8(100));
    mmshow(b);
    c = mmlabel(b);
    mmlblshow(c,'border');
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
    d = mmblob(c,'area');
    mmshow(d);
    mmshow(d, mmcmp(d,'==',0));'''
    d = mmblob(c,'area');
    mmshow(d);
    mmshow(d, mmcmp(d,'==',0));
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
    wings = mmcmp( uint16(100),'<=',d, '<=', uint16(2500));
    mmshow(wings);
    tights = mmcmp( uint16(2500),'<',d, '<=', uint16(5500));
    mmshow(tights);'''
    wings = mmcmp( uint16(100),'<=',d, '<=', uint16(2500));
    mmshow(wings);
    tights = mmcmp( uint16(2500),'<',d, '<=', uint16(5500));
    mmshow(tights);
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
    legs = mmcmp( uint16(5500), '<', d, '<=', uint16(8500));
    mmshow(legs);
    breast = mmcmp( d,'>', uint16(8500));
    mmshow(breast);'''
    legs = mmcmp( uint16(5500), '<', d, '<=', uint16(8500));
    mmshow(legs);
    breast = mmcmp( d,'>', uint16(8500));
    mmshow(breast);
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
    mmshow(a, mmgradm(wings), mmgradm(tights), mmgradm(legs),mmgradm(breast));'''
    mmshow(a, mmgradm(wings), mmgradm(tights), mmgradm(legs),mmgradm(breast));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdconcrete - Aggregate and anhydrous phase extraction from a concrete section observed by a SEM image.
#
# =========================================================================
def mmdconcrete():

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
    f = mmreadgray('csample.jpg')
    mmshow(f)'''
    f = mmreadgray('csample.jpg')
    mmshow(f)
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
    h = mmhistogram(f)
    mmplot([[h]])'''
    h = mmhistogram(f)
    mmplot([[h]])
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
    hf = mmasf(mmneg(h),'co',mmseline(5))
    ws = mmwatershed(hf)
    wsf = mmintersec(mmneg(mmframe(ws,20)),ws)
    t = nonzero(wsf)
    print t
    mmax = mmstats(h,'max')
    hf_plot = mmneg(hf)
    ws_plot = mmgray(ws, 'uint16', mmax)
    wsf_plot = mmgray(wsf, 'uint16', mmax)
    mmplot([[hf_plot],[ws_plot],[wsf_plot]])'''
    hf = mmasf(mmneg(h),'co',mmseline(5))
    ws = mmwatershed(hf)
    wsf = mmintersec(mmneg(mmframe(ws,20)),ws)
    t = nonzero(wsf)
    print t
    mmax = mmstats(h,'max')
    hf_plot = mmneg(hf)
    ws_plot = mmgray(ws, 'uint16', mmax)
    wsf_plot = mmgray(wsf, 'uint16', mmax)
    mmplot([[hf_plot],[ws_plot],[wsf_plot]])
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
    aux = mmthreshad( f, t, 255)
    anidro = mmareaopen(aux, 20)
    mmshow( f, mmgradm(anidro))'''
    aux = mmthreshad( f, t, 255)
    anidro = mmareaopen(aux, 20)
    mmshow( f, mmgradm(anidro))
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
    g=mmgradm(f)
    m=mmregmin(mmhmin(g,10))
    ws=mmcwatershed(g,m)
    mmshow(ws)'''
    g=mmgradm(f)
    m=mmregmin(mmhmin(g,10))
    ws=mmcwatershed(g,m)
    mmshow(ws)
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
    aux1=mmareaopen(mmneg(ws),300)
    aux2=mmareaclose(aux1,50)
    aggr=mmsubm(aux2,anidro)
    mmshow(f, mmgradm(aggr))'''
    aux1=mmareaopen(mmneg(ws),300)
    aux2=mmareaclose(aux1,50)
    aggr=mmsubm(aux2,anidro)
    mmshow(f, mmgradm(aggr))
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
    anidro_phase = mmstats(anidro,'sum')/n
    print 'anidro=',anidro_phase
    aggr_phase = mmstats(aggr,'sum')/n;
    print 'aggr=',aggr_phase
    mmshow( f, mmgradm(aggr), mmgradm(anidro))'''
    n = product(shape(f))
    anidro_phase = mmstats(anidro,'sum')/n
    print 'anidro=',anidro_phase
    aggr_phase = mmstats(aggr,'sum')/n;
    print 'aggr=',aggr_phase
    mmshow( f, mmgradm(aggr), mmgradm(anidro))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdcookies - Detect broken rounded biscuits.
#
# =========================================================================
def mmdcookies():

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
    a = mmreadgray('cookies.tif');
    mmshow(a);'''
    a = mmreadgray('cookies.tif');
    mmshow(a);
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
    b = mmthreshad(a, uint8(100));
    mmshow(b);'''
    b = mmthreshad(a, uint8(100));
    mmshow(b);
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
    c = mmopenth(b,mmsedisk(55,'2D','OCTAGON'));
    mmshow(c);'''
    c = mmopenth(b,mmsedisk(55,'2D','OCTAGON'));
    mmshow(c);
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
    d = mmareaopen(c,400);
    mmshow(d);'''
    d = mmareaopen(c,400);
    mmshow(d);
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
    mmshow(a,d);'''
    mmshow(a,d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdcornea - Cornea cells marking.
#
# =========================================================================
def mmdcornea():

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
    a = mmreadgray('corneacells.tif');
    mmshow(a);
    b = mmsurf(a);
    mmshow(b);'''
    a = mmreadgray('corneacells.tif');
    mmshow(a);
    b = mmsurf(a);
    mmshow(b);
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
    c = mmasf(a,'oc',mmsecross(),2);
    d = mmregmax( c);
    mmshow(mmsurf(c));
    mmshow(mmsurf(c), d);'''
    c = mmasf(a,'oc',mmsecross(),2);
    d = mmregmax( c);
    mmshow(mmsurf(c));
    mmshow(mmsurf(c), d);
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
    e = mmneg(a);
    f = mmcwatershed(e, d, mmsebox());
    mmshow(e,d);
    mmshow(e,f,d);'''
    e = mmneg(a);
    f = mmcwatershed(e, d, mmsebox());
    mmshow(e,d);
    mmshow(e,f,d);
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
    g = mmgray(f, 'uint16', 1);
    h1 = mmaddm(mmlabel(d), uint16(1));
    h = mmintersec(mmgray(d,'uint16'), h1);
    i = mmunion( g, h);
    mmlblshow(i);
    j = mmgradm( a);
    mmshow(j);'''
    g = mmgray(f, 'uint16', 1);
    h1 = mmaddm(mmlabel(d), uint16(1));
    h = mmintersec(mmgray(d,'uint16'), h1);
    i = mmunion( g, h);
    mmlblshow(i);
    j = mmgradm( a);
    mmshow(j);
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
    k = mmcwatershed(j, i);
    mmshow( k);
    mmshow(a, k, k);'''
    k = mmcwatershed(j, i);
    mmshow( k);
    mmshow(a, k, k);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdfabric - Detection of vertical weave in fabrics.
#
# =========================================================================
def mmdfabric():

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
    a = mmreadgray('fabric.tif');
    mmshow(a);'''
    a = mmreadgray('fabric.tif');
    mmshow(a);
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
    b = mmdil(a,mmsedisk(4));
    c = mmdil(b,mmseline(25,90));
    d = mmdil(c,mmseline(25,-90));
    mmshow(d);'''
    b = mmdil(a,mmsedisk(4));
    c = mmdil(b,mmseline(25,90));
    d = mmdil(c,mmseline(25,-90));
    mmshow(d);
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
    e = mmhmin(d,25);
    f = mmregmin(e);
    mmshow(f);'''
    e = mmhmin(d,25);
    f = mmregmin(e);
    mmshow(f);
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
    g = mmcwatershed(a,f);
    mmshow(a,mmdil(g));'''
    g = mmcwatershed(a,f);
    mmshow(a,mmdil(g));
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
    h = mmunion(g,f);
    i = mmgradm(a);
    mmshow(h);
    mmshow(i);'''
    h = mmunion(g,f);
    i = mmgradm(a);
    mmshow(h);
    mmshow(i);
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
    j = mmcwatershed(i,h,mmsebox());
    mmshow(a,j);'''
    j = mmcwatershed(i,h,mmsebox());
    mmshow(a,j);
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
    k = mmcwatershed(i,h,mmsebox(),'REGIONS');
    mmlblshow(k,'border');
    l = mmblob(k,'area');
    mmshow(l);'''
    k = mmcwatershed(i,h,mmsebox(),'REGIONS');
    mmlblshow(k,'border');
    l = mmblob(k,'area');
    mmshow(l);
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
    m = mmcmp(l,'<=',2000);
    mmshow(m);
    n = mmlabel(m,mmsebox());
    mmlblshow(n,'border');'''
    m = mmcmp(l,'<=',2000);
    mmshow(m);
    n = mmlabel(m,mmsebox());
    mmlblshow(n,'border');
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
    mmshow(a);
    o = mmblob(n,'area','data');
    mmplot([[o]],[['style','impulses']])'''
    mmshow(a);
    o = mmblob(n,'area','data');
    mmplot([[o]],[['style','impulses']])
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdfila - Detect Filarial Worms.
#
# =========================================================================
def mmdfila():

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
    a = mmreadgray('danaus.tif');
    mmshow(a);'''
    a = mmreadgray('danaus.tif');
    mmshow(a);
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
    b = mmcloserecth(a,mmsebox(5));
    mmshow(b);'''
    b = mmcloserecth(a,mmsebox(5));
    mmshow(b);
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
    c = mmopen(b);
    mmshow(c);'''
    c = mmopen(b);
    mmshow(c);
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
    d = mmareaopen(c,200);
    mmshow(d);'''
    d = mmareaopen(c,200);
    mmshow(d);
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
    e = mmthreshad(d,50);
    mmshow(e);'''
    e = mmthreshad(d,50);
    mmshow(e);
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
    f = mmthin(e);
    mmshow(f);'''
    f = mmthin(e);
    mmshow(f);
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
    g = mmthin(f,mmendpoints(), 12);
    mmshow(g);'''
    g = mmthin(f,mmendpoints(), 12);
    mmshow(g);
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
    h = mminfrec(g,e);
    mmshow(h);'''
    h = mminfrec(g,e);
    mmshow(h);
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
    mmshow(a,h);'''
    mmshow(a,h);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdflatzone - Flat-zone image simplification by connected filtering.
#
# =========================================================================
def mmdflatzone():

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
    a = mmreadgray('cameraman.tif')
    mmshow(a)'''
    a = mmreadgray('cameraman.tif')
    mmshow(a)
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
    b = mmlabelflat(a,mmsebox())
    nfz=mmstats(b,'max')
    print nfz
    mmshow(a)
    mmlblshow(b)'''
    b = mmlabelflat(a,mmsebox())
    nfz=mmstats(b,'max')
    print nfz
    mmshow(a)
    mmlblshow(b)
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
    c=mmasfrec(a,'CO',mmsebox(),mmsebox(),2)
    d=mmasfrec(a,'CO',mmsebox(),mmsebox(),4)
    e=mmasfrec(a,'CO',mmsebox(),mmsebox(),16)
    mmshow(c)
    mmshow(d)
    mmshow(e)'''
    c=mmasfrec(a,'CO',mmsebox(),mmsebox(),2)
    d=mmasfrec(a,'CO',mmsebox(),mmsebox(),4)
    e=mmasfrec(a,'CO',mmsebox(),mmsebox(),16)
    mmshow(c)
    mmshow(d)
    mmshow(e)
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
    c_lab=mmlabelflat(c,mmsebox())
    d_lab=mmlabelflat(d,mmsebox())
    e_lab=mmlabelflat(e,mmsebox())
    print mmstats(c_lab,'max')
    print mmstats(d_lab,'max')
    print mmstats(e_lab,'max')
    mmlblshow(c_lab)
    mmlblshow(d_lab)
    mmlblshow(e_lab)'''
    c_lab=mmlabelflat(c,mmsebox())
    d_lab=mmlabelflat(d,mmsebox())
    e_lab=mmlabelflat(e,mmsebox())
    print mmstats(c_lab,'max')
    print mmstats(d_lab,'max')
    print mmstats(e_lab,'max')
    mmlblshow(c_lab)
    mmlblshow(d_lab)
    mmlblshow(e_lab)
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
    c_flat=mmcmp(c_lab,'==',c_v)
    d_v=d_lab[89,59]
    d_flat=mmcmp(d_lab,'==',d_v)
    e_v=e_lab[89,59]
    e_flat=mmcmp(e_lab,'==',e_v)
    mmshow(a,e_flat,d_flat,c_flat)'''
    c_v=c_lab[89,59]
    c_flat=mmcmp(c_lab,'==',c_v)
    d_v=d_lab[89,59]
    d_flat=mmcmp(d_lab,'==',d_v)
    e_v=e_lab[89,59]
    e_flat=mmcmp(e_lab,'==',e_v)
    mmshow(a,e_flat,d_flat,c_flat)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdflow - Detect water in a static image of an oil-water flow experiment.
#
# =========================================================================
def mmdflow():

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
    a = mmreadgray('flow.tif')
    mmshow(a)'''
    a = mmreadgray('flow.tif')
    mmshow(a)
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
    b=mmcloseth(a,mmseline(50,90))
    mmshow(b)'''
    b=mmcloseth(a,mmseline(50,90))
    mmshow(b)
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
    c=mmcloserec(b,mmsebox(5))
    mmshow(c)'''
    c=mmcloserec(b,mmsebox(5))
    mmshow(c)
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
    d=mmasf(c,'co',mmsecross())
    mmshow(d)'''
    d=mmasf(c,'co',mmsecross())
    mmshow(d)
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
    e=mmthreshad(d,100)
    mmshow(a)
    mmshow(a,e)'''
    e=mmthreshad(d,100)
    mmshow(a)
    mmshow(a,e)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdgear - Detect the teeth of a gear
#
# =========================================================================
def mmdgear():

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
    a = mmreadgray('gear.tif');
    mmshow(a);'''
    a = mmreadgray('gear.tif');
    mmshow(a);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Opening of the input image by an Euclidean disk of radius 20. The
    sequence opening-subtraction is called opening top-hat. The opening
    top-hat could be executed in a single command: c = (a,(20));
    '''
    print '========================================================================='
    #0
    print '''
    b = mmopen(a,mmsedisk(20));
    mmshow(b);
    c = mmsubm(a,b);
    mmshow(c);'''
    b = mmopen(a,mmsedisk(20));
    mmshow(b);
    c = mmsubm(a,b);
    mmshow(c);
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
    d = mmlabel(c);
    nteeth=mmstats(d,'max')
    mmlblshow(d,'border');'''
    d = mmlabel(c);
    nteeth=mmstats(d,'max')
    mmlblshow(d,'border');
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdholecenter - Hole center misalignment in PCB.
#
# =========================================================================
def mmdholecenter():

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
    a = mmreadgray('pcbholes.tif')
    mmshow(a)'''
    a = mmreadgray('pcbholes.tif')
    mmshow(a)
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
    b = mmclohole(a)
    mmshow(b)
    d = mmdist(b,mmsecross(),'EUCLIDEAN')
    e = mmregmax(d,mmsebox())
    f = mmthreshad(d, uint16([20]))   # radius larger than 20 pixels
    g = mmintersec(e,f)
    h = mmblob(mmlabel(g,mmsebox()),'CENTROID'); # pad center
    mmshow(b,mmdil(h))'''
    b = mmclohole(a)
    mmshow(b)
    d = mmdist(b,mmsecross(),'EUCLIDEAN')
    e = mmregmax(d,mmsebox())
    f = mmthreshad(d, uint16([20]))   # radius larger than 20 pixels
    g = mmintersec(e,f)
    h = mmblob(mmlabel(g,mmsebox()),'CENTROID'); # pad center
    mmshow(b,mmdil(h))
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
    i = mmsubm(b,a)
    mmshow(i)
    j = mmdist(i,mmsecross(),'EUCLIDEAN')
    k = mmregmax(j,mmsebox())
    l = mmblob(mmlabel(k,mmsebox()),'CENTROID') # hole center
    mmshow(i,mmdil(l))'''
    i = mmsubm(b,a)
    mmshow(i)
    j = mmdist(i,mmsecross(),'EUCLIDEAN')
    k = mmregmax(j,mmsebox())
    l = mmblob(mmlabel(k,mmsebox()),'CENTROID') # hole center
    mmshow(i,mmdil(l))
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
    mmshow(a,h,l)
    m = mmdist(mmneg(l),mmsecross(),'EUCLIDEAN')
    n = mmintersec(mmgray(h),uint8(m))
    mmshow(n,a)
    i = nonzero(n.ravel())
    x = i / n.shape[1]
    y = i % n.shape[1]
    for k in range(len(i)):
      print 'displacement of %d at (%d,%d)\n' %(n[x[k],y[k]],x[k],y[k])'''
    from numpy.oldnumeric import nonzero
    mmshow(a,h,l)
    m = mmdist(mmneg(l),mmsecross(),'EUCLIDEAN')
    n = mmintersec(mmgray(h),uint8(m))
    mmshow(n,a)
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
    o=mmthin(a)
    p=mmthin(o,mmendpoints())
    mmshow(a,p)
    q = mmdist(a,mmsecross(),'EUCLIDEAN')
    r = mmgrain(mmlabel(p,mmsebox()),q,'min') # minimum
    s = mmgrain(mmlabel(p,mmsebox()),q,'min','data') # minimum
    from numpy.oldnumeric import ravel
    for k in ravel(s):
      print 'Minimum distance: %d pixels' %(2*k+1)
    t = mmintersec(mmcmp(r,'==',q),a)
    mmshow(a,mmdil(t))'''
    o=mmthin(a)
    p=mmthin(o,mmendpoints())
    mmshow(a,p)
    q = mmdist(a,mmsecross(),'EUCLIDEAN')
    r = mmgrain(mmlabel(p,mmsebox()),q,'min') # minimum
    s = mmgrain(mmlabel(p,mmsebox()),q,'min','data') # minimum
    from numpy.oldnumeric import ravel
    for k in ravel(s):
      print 'Minimum distance: %d pixels' %(2*k+1)
    t = mmintersec(mmcmp(r,'==',q),a)
    mmshow(a,mmdil(t))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdleaf - Segment a leaf from the background
#
# =========================================================================
def mmdleaf():

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
    f = mmreadgray('leaf.tif')
    mmshow(f)'''
    f = mmreadgray('leaf.tif')
    mmshow(f)
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
    f_low=mmthreshad(f,100)
    f_med=mmthreshad(f,128)
    f_high=mmthreshad(f,160)
    mmshow(f_low)
    mmshow(f_med)
    mmshow(f_high)'''
    f_low=mmthreshad(f,100)
    f_med=mmthreshad(f,128)
    f_high=mmthreshad(f,160)
    mmshow(f_low)
    mmshow(f_med)
    mmshow(f_high)
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
    f1=mmtoggle(f,mmero(f,mmsedisk(7)),mmdil(f,mmsedisk(7)))
    mmshow(f1)'''
    f1=mmtoggle(f,mmero(f,mmsedisk(7)),mmdil(f,mmsedisk(7)))
    mmshow(f1)
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
    f2=mmthreshad(f1,100)
    f3=mmareaopen(f2,80)
    mmshow(f2)
    mmshow(f3)'''
    f2=mmthreshad(f1,100)
    f3=mmareaopen(f2,80)
    mmshow(f2)
    mmshow(f3)
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
    f4=mmgradm(f3)
    mmshow(f,f4)'''
    f4=mmgradm(f3)
    mmshow(f,f4)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdlith - Detect defects in a microelectronic circuit.
#
# =========================================================================
def mmdlith():

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
    a = mmreadgray('r4x2_256.tif');
    mmshow(a);
    mmshow(mmsurf(a));'''
    a = mmreadgray('r4x2_256.tif');
    mmshow(a);
    mmshow(mmsurf(a));
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Closing of the image by a vertical line of length 25 pixels. Then
    subtract it from the original. The sequence closing-subtraction is
    called closing top-hat. (This could be executed in a single command:
    c=(a,(25,90));).
    '''
    print '========================================================================='
    #0
    print '''
    b = mmclose(a,mmseline(25,90));
    mmshow(b);
    mmshow(mmsurf(b));'''
    b = mmclose(a,mmseline(25,90));
    mmshow(b);
    mmshow(mmsurf(b));
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
    c = mmsubm(b,a);
    mmshow(c);
    mmshow(mmsurf(c));'''
    c = mmsubm(b,a);
    mmshow(c);
    mmshow(mmsurf(c));
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
    d = mmcmp(c,'>=',50);
    e = mmareaopen(d,5);
    mmshow(d);
    mmshow(e);'''
    d = mmcmp(c,'>=',50);
    e = mmareaopen(d,5);
    mmshow(d);
    mmshow(e);
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
    mmshow(a,e);
    mmshow(mmsurf(a),e);'''
    mmshow(a,e);
    mmshow(mmsurf(a),e);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdpcb - Decompose a printed circuit board in its main parts.
#
# =========================================================================
def mmdpcb():

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
    a = mmreadgray('pcb1bin.tif');
    mmshow(a);'''
    a = mmreadgray('pcb1bin.tif');
    mmshow(a);
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
    b = mmclohole(a);
    holes = mmsubm(b,a);
    mmshow(b);
    mmshow(a, holes);'''
    b = mmclohole(a);
    holes = mmsubm(b,a);
    mmshow(b);
    mmshow(a, holes);
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
    c = mmopen(b,mmsebox(8));
    square = mmcdil(c, a);
    mmshow(b, c);
    mmshow(holes, square);'''
    c = mmopen(b,mmsebox(8));
    square = mmcdil(c, a);
    mmshow(b, c);
    mmshow(holes, square);
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
    f = mmsubm(b, c);
    g = mmopen(f, mmsedisk(8));
    circle = mmcdil(g,a);
    mmshow(f, g);
    mmshow(holes, square, circle);'''
    f = mmsubm(b, c);
    g = mmopen(f, mmsedisk(8));
    circle = mmcdil(g,a);
    mmshow(f, g);
    mmshow(holes, square, circle);
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
    i = mmsubm(f, g);
    m = mmopen(i,mmsedil(mmseline(8,90), mmseline(25)));
    rect = mmcdil(m,a);
    mmshow(i, m);
    mmshow(holes, square, circle, rect);'''
    i = mmsubm(f, g);
    m = mmopen(i,mmsedil(mmseline(8,90), mmseline(25)));
    rect = mmcdil(m,a);
    mmshow(i, m);
    mmshow(holes, square, circle, rect);
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
    o = mmsubm(i,m);
    p = mmopen(o, mmsebox(2));
    thin = mmcdil(p,a);
    mmshow(o, p);
    mmshow(holes, square, circle, rect, thin);'''
    o = mmsubm(i,m);
    p = mmopen(o, mmsebox(2));
    thin = mmcdil(p,a);
    mmshow(o, p);
    mmshow(holes, square, circle, rect, thin);
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
    r = mmsubm(o,p);
    s = mmopen(r, mmsebox());
    thick = mmcdil(s,a);
    mmshow(r, s);
    mmshow(holes, square, circle, rect, thin, thick);'''
    r = mmsubm(o,p);
    s = mmopen(r, mmsebox());
    thick = mmcdil(s,a);
    mmshow(r, s);
    mmshow(holes, square, circle, rect, thin, thick);
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
    mmshow(holes, square, circle, rect, thin, thick);'''
    mmshow(holes, square, circle, rect, thin, thick);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdpieces - Classify two dimensional pieces.
#
# =========================================================================
def mmdpieces():

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
    a = mmreadgray('pieces_bw.tif');
    mmshow(a);'''
    a = mmreadgray('pieces_bw.tif');
    mmshow(a);
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
    seA = mmimg2se(mmbinary([[0,1,0],[1,0,1],[0,0,0]]))
    seB = mmimg2se(mmbinary([[0,0,0],[0,1,0],[0,0,0]]))
    iAB = mmse2hmt(seA,seB);
    print mmintershow(iAB)
    b = mmthick(a, iAB);
    mmshow(b);'''
    seA = mmimg2se(mmbinary([[0,1,0],[1,0,1],[0,0,0]]))
    seB = mmimg2se(mmbinary([[0,0,0],[0,1,0],[0,0,0]]))
    iAB = mmse2hmt(seA,seB);
    print mmintershow(iAB)
    b = mmthick(a, iAB);
    mmshow(b);
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
    c = mmthin(b);
    mmshow(c);'''
    c = mmthin(b);
    mmshow(c);
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
    d = mmthin(c,mmendpoints());
    mmshow(c,d);'''
    d = mmthin(c,mmendpoints());
    mmshow(c,d);
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
    e = mminfrec(d,b);
    mmshow(e);'''
    e = mminfrec(d,b);
    mmshow(e);
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
    f = mmintersec(a,e);
    mmshow(f);'''
    f = mmintersec(a,e);
    mmshow(f);
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
    g = mmsubm(c,e);
    mmshow(g);'''
    g = mmsubm(c,e);
    mmshow(g);
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
    h = mmthin(g, mmendpoints(), 4);
    mmshow(h);'''
    h = mmthin(g, mmendpoints(), 4);
    mmshow(h);
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
    seA1 = mmimg2se(mmbinary([[0,1,0],[0,1,0],[1,0,1]]))
    seB1 = mmimg2se(mmbinary([[0,0,0],[1,0,1],[0,1,0]]))
    seA2 = mmimg2se(mmbinary([[0,1,0],[1,1,1],[0,0,0]]))
    seB2 = mmimg2se(mmbinary([[1,0,1],[0,0,0],[0,1,0]]))
    i1 = mmsupcanon(h, mmse2hmt(seA1,seB1));
    i2 = mmsupcanon(h, mmse2hmt(seA2,seB2));
    i = mmunion(i1,i2);
    mmshow(h,mmdil(i,mmsedisk(2)));'''
    seA1 = mmimg2se(mmbinary([[0,1,0],[0,1,0],[1,0,1]]))
    seB1 = mmimg2se(mmbinary([[0,0,0],[1,0,1],[0,1,0]]))
    seA2 = mmimg2se(mmbinary([[0,1,0],[1,1,1],[0,0,0]]))
    seB2 = mmimg2se(mmbinary([[1,0,1],[0,0,0],[0,1,0]]))
    i1 = mmsupcanon(h, mmse2hmt(seA1,seB1));
    i2 = mmsupcanon(h, mmse2hmt(seA2,seB2));
    i = mmunion(i1,i2);
    mmshow(h,mmdil(i,mmsedisk(2)));
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
    j = mminfrec(i,b,mmsebox());
    mmshow(j);'''
    j = mminfrec(i,b,mmsebox());
    mmshow(j);
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
    k = mmintersec(a,j);
    mmshow(k);'''
    k = mmintersec(a,j);
    mmshow(k);
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
    l = mmsubm(mmsubm(a,f),k);
    mmshow(l);'''
    l = mmsubm(mmsubm(a,f),k);
    mmshow(l);
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
    m = mmgray(f,'uint8',1);
    n = mmgray(k,'uint8',2);
    o = mmgray(l,'uint8',3);
    p = mmunion(m,n,o);
    mmlblshow(p);'''
    m = mmgray(f,'uint8',1);
    n = mmgray(k,'uint8',2);
    o = mmgray(l,'uint8',3);
    p = mmunion(m,n,o);
    mmlblshow(p);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdpotatoes - Grade potato quality by shape and skin spots.
#
# =========================================================================
def mmdpotatoes():

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
    a = mmreadgray('potatoes.tif');
    mmshow(a);'''
    a = mmreadgray('potatoes.tif');
    mmshow(a);
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
    b = mmthreshad(a,90);
    mmshow(b);'''
    b = mmthreshad(a,90);
    mmshow(b);
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
    c = mmthin(b);
    mmshow(a,c);'''
    c = mmthin(b);
    mmshow(a,c);
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
    d = mmcloseth(a,mmsedisk(5));
    mmshow(d);'''
    d = mmcloseth(a,mmsedisk(5));
    mmshow(d);
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
    e = mmthreshad(d,20);
    f = mmintersec(e,b);
    mmshow(f);'''
    e = mmthreshad(d,20);
    f = mmintersec(e,b);
    mmshow(f);
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
    mmshow(a);
    mmshow(a,f,c);'''
    mmshow(a);
    mmshow(a,f,c);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdrobotop - Detect marks on a robot.
#
# =========================================================================
def mmdrobotop():

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
    a = mmreadgray('robotop.tif');
    mmshow(a);'''
    a = mmreadgray('robotop.tif');
    mmshow(a);
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
    b = mmopenth(a,mmsebox(4));
    mmshow(b);'''
    b = mmopenth(a,mmsebox(4));
    mmshow(b);
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
    c = mmopen(b,mmsebox());
    mmshow(c);'''
    c = mmopen(b,mmsebox());
    mmshow(c);
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
    d = mmthreshad(c,100);
    mmshow(a,d);'''
    d = mmthreshad(c,100);
    mmshow(a,d);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdruler - Detect defects in a ruler.
#
# =========================================================================
def mmdruler():

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
    a = mmreadgray('mm3.tif');
    mmshow(a);'''
    a = mmreadgray('mm3.tif');
    mmshow(a);
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
    b = mmthreshad( mmcloseth(a,mmsebox(5)),40);
    mmshow(b);'''
    b = mmthreshad( mmcloseth(a,mmsebox(5)),40);
    mmshow(b);
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
    c = mmopen(b,mmseline(50,90));
    mmshow(c);'''
    c = mmopen(b,mmseline(50,90));
    mmshow(c);
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
    d =mmclose(c,mmseline(15));
    mmshow(d);'''
    d =mmclose(c,mmseline(15));
    mmshow(d);
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
    e = mminfrec(d,b);
    mmshow(e);'''
    e = mminfrec(d,b);
    mmshow(e);
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
    f = mminfrec(d,b,mmseline(3,90));
    mmshow(f);'''
    f = mminfrec(d,b,mmseline(3,90));
    mmshow(f);
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
    g = mmsubm(e,f);
    mmshow(g);'''
    g = mmsubm(e,f);
    mmshow(g);
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
    h = mmopen(g);
    mmshow(h);'''
    h = mmopen(g);
    mmshow(h);
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
    i = mminfrec(h, b);
    mmshow(i);'''
    i = mminfrec(h, b);
    mmshow(i);
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
    mmshow(a,i);'''
    mmshow(a,i);
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   mmdsoil - Detect fractures in soil.
#
# =========================================================================
def mmdsoil():

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
    a = mmreadgray('soil.tif');
    mmshow(a);'''
    a = mmreadgray('soil.tif');
    mmshow(a);
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
    b = mmcloseth(a,mmsebox(2));
    mmshow(b);'''
    b = mmcloseth(a,mmsebox(2));
    mmshow(b);
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
    c= mmareaopen(b,80,mmsebox());
    mmshow(c);'''
    c= mmareaopen(b,80,mmsebox());
    mmshow(c);
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
    d = mmthreshad(c,15);
    mmshow(d);'''
    d = mmthreshad(c,15);
    mmshow(d);
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
    mmshow(a,d);'''
    mmshow(a,d);
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
_alldemos = []

_alldemos.append('mmdairport') 
_alldemos.append('mmdarea') 
_alldemos.append('mmdasp') 
_alldemos.append('mmdlabeltext') 
_alldemos.append('mmdbeef') 
_alldemos.append('mmdblob') 
_alldemos.append('mmdbrain') 
_alldemos.append('mmdcalc') 
_alldemos.append('mmdcells') 
_alldemos.append('mmdchickparts') 
_alldemos.append('mmdconcrete') 
_alldemos.append('mmdcookies') 
_alldemos.append('mmdcornea') 
_alldemos.append('mmdfabric') 
_alldemos.append('mmdfila') 
_alldemos.append('mmdflatzone') 
_alldemos.append('mmdflow') 
_alldemos.append('mmdgear') 
_alldemos.append('mmdholecenter') 
_alldemos.append('mmdleaf') 
_alldemos.append('mmdlith') 
_alldemos.append('mmdpcb') 
_alldemos.append('mmdpieces') 
_alldemos.append('mmdpotatoes') 
_alldemos.append('mmdrobotop') 
_alldemos.append('mmdruler') 
_alldemos.append('mmdsoil') 

#main execution
print '\nmorph Demonstrations -- SDC Morphology Toolbox\n'
print 'Available Demonstrations: \n' + str(_alldemos) + '\n'
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for demo in sys.argv[1:]:
            if _alldemos.count(demo) > 0:
                eval(demo + '()')
            else:
                print "Demonstration " + demo + " is not in this package. Please use help for details\n"
    else:
        print "\nUsage: python morph.py <demo_name>\n\n"
else:
    print 'Please use help(morphdemo) for details\n'

