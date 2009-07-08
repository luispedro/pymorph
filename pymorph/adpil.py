"""
    Module adpil -- Toolbox adpil
    -------------------------------------------------------------------
    This module provides a link between numpy arrays and PIL, Python Imaging
    Library, images. Its functions perform image file I/O (in formats supported
    by PIL) and displaying of images represented as numpy arrays. The layout
    of these numpy arrays follows the rules of the adimage toolbox images.
    -------------------------------------------------------------------
    adimages()    -- List image files located on sys.imagepath, if this variable
                     exists, or, otherwise, on sys.path
    adread()      -- Read an image from a file to a numpy array.
    adshow()      -- Display an image
    adshowclear() -- Close all adshow windows.
    adshowfile()  -- Display an image file
    adshowmode()  -- Set/get the current operational mode.
    adwrite()     -- Write an image from a numpy array to an image file. The
                     format is deduced from the filename extension.
    array2pil()   -- Convert a numpy array to a PIL image
    pil2array()   -- Convert a PIL image to a numpy array

"""

import warnings
warnings.warn('''\
pymorph.adpil is deprecated and un-maintained (with lurking bugs).

Consider the following (more feature-full) alternatives:
    * matplotlib to display images (http://matplotlib.sourceforge.net/)
    * readmagick (http://pypi.python.org/pypi/readmagick)
        or scipy.misc.pilutil to read/write images to/from disk.

''', DeprecationWarning)

__version__ = '1.0 all'
__version_string__ = 'Toolbox adpil V1.0 28Jul2003'
__build_date__ = '04aug2003 11:29'

def findImageFile(filename):
    '''Search image filename in sys.imagepath or sys.path.'''
    import sys, os.path
    if not os.path.isfile(filename) and not os.path.isabs(filename):
        try:
            for a in sys.imagepath:
                if os.path.isfile(os.path.join(a, filename)):
                    filename = os.path.join(a, filename)
                    break
        except:
            for a in sys.path:
                if os.path.isfile(os.path.join(a, filename)):
                    filename = os.path.join(a, filename)
                    break
    return filename


def adread(imagefile):
    """
        - Purpose
            Read an image from a file to a numpy array.
        - Synopsis
            arr = adread(imagefile)
        - Input
            imagefile: Image file path.
        - Output
            arr: numpy array representing an image.

    """

    import Image
    img = findImageFile(imagefile)
    arr = pil2array(Image.open(img))
    return arr


def adwrite(imagefile, arr):
    """
        - Purpose
            Write an image from a numpy array to an image file. The format
            is deduced from the filename extension.
        - Synopsis
            adwrite(imagefile, arr)
        - Input
            imagefile: Image file path.
            arr:       The numpy array to save.

    """

    array2pil(arr).save(imagefile)
    return


def listImageFiles(glb='*'):
    '''List image files located on sys.path.'''
    import sys, os.path, glob
    if os.path.splitext(glb)[1] == '':
        imgexts = ['.tif', '.jpg', '.gif', '.png', '.pbm', '.pgm', '.ppm', '.bmp']
    else:
        imgexts = ['']
    images = {}
    try:
        for dir in sys.imagepath:
            for ext in imgexts:
                for ff in glob.glob(os.path.join(dir, glb + ext)):
                    images[os.path.basename(ff)] = ff
    except:
        for dir in sys.path:
            for ext in imgexts:
                for ff in glob.glob(os.path.join(dir, glb + ext)):
                    images[os.path.basename(ff)] = ff
    return images


def adimages(glob='*'):
    """
        - Purpose
            List image files located on sys.imagepath, if this variable
            exists, or, otherwise, on sys.path
        - Synopsis
            imglist = adimages(glob='*')
        - Input
            glob: Default: '*'. Glob string for the image filename.
        - Output
            imglist: Image filename list.

    """

    lst = listImageFiles(glob).keys()
    lst.sort()
    return lst
    return imglist



import Tkinter
import Image, ImageTk
def tkRoot ():
    '''Returns the current Tk root.'''
    if Tkinter._default_root is None:
        root = Tkinter.Tk()
        Tkinter._default_root.withdraw()
    else:
        root = Tkinter._default_root
    return root
tk_root = None      # posterga a ativacao do Tk ateh que adshow seja chamado
show_mode = 0       # qdo 1, cria sempre um novo viewer

class ImageViewer(Tkinter.Toplevel):
    '''The implementation base class for adshow.'''
    viewmap = {}
    geomap = {}
    def __init__(self, id):
        self.image = None
        self.id = id
        ImageViewer.viewmap[id] = self
        Tkinter.Toplevel.__init__(self)
        self.protocol("WM_DELETE_WINDOW", self.done)
    def show(self, arr, title=None):
        if self.image is not None:
            self.image.pack_forget()
        if title is not None:
            self.title(title)
        self.image = self.getTkLabel(arr)
        self.image.pack(fill='both', expand=1)
        self.image.tkraise()
        if self.geomap.get(self.id):
            self.geometry(self.geomap[self.id])
        self.tkraise()
    def getTkLabel(self, arr):
        self.PILimage = array2pil(arr)
        self.TKimage = ImageTk.PhotoImage(self.PILimage)
        return Tkinter.Label(self, image=self.TKimage, bg='gray', bd=0)
    def done(self):
        self.geomap[self.id] = self.geometry()
        del ImageViewer.viewmap[self.id]
        self.destroy()


def adshow(arr, title='adshow', id=0):
    """
        - Purpose
            Display an image
        - Synopsis
            adshow(arr, title='adshow', id=0)
        - Input
            arr:   The numpy array to display.
            title: Default: 'adshow'. Title of the view window.
            id:    Default: 0. An identification for the window.

        - Description
            Display an image from a numpy array in a Tk toplevel with
            title given by argument 'title'.

    """

    import numpy
    global tk_root
    global show_mode
    if tk_root is None:
        # ativa Tk
        tk_root = tkRoot()
    if arr.dtype.char == '1':
        arr = numpy.array(arr*255).astype('B')
    if show_mode:
        vw = ImageViewer(len(ImageViewer.viewmap.keys()))
    elif ImageViewer.viewmap.get(id):
        vw = ImageViewer.viewmap[id]
    else:
        vw = ImageViewer(id)
    vw.show(arr, title)
    return


def adshowmode(newmode=None):
    """
        - Purpose
            Set/get the current operational mode.
        - Synopsis
            mode = adshowmode(newmode=None)
        - Input
            newmode: Default: None. New operational mode. If None, returns
                     the current mode. Mode values are 0: the adshow arg
                     'id' identifies the window where to display the image;
                     1: create a new window for each image.
        - Output
            mode: Current mode.

    """

    global show_mode
    if newmode:
        show_mode = newmode
    return show_mode


def adshowclear():
    """
        - Purpose
            Close all adshow windows.
        - Synopsis
            adshowclear()
    """
    for id in ImageViewer.viewmap.keys():
        ImageViewer.viewmap[id].done()


def adshowfile(filepath, id=0):
    """
        - Purpose
            Display an image file
        - Synopsis
            adshowfile(filepath, id=0)
        - Input
            filepath: Image file path.
            id:       Default: 0. An identification for the window.

        - Description
            Display an image file. Uses adshow(). The title is the tail of
            the filename. Argument 'id' is the same as adshow().

    """

    import os.path
    path = findImageFile(filepath)
    img = adread(path)
    adshow(img, os.path.basename(filepath), id)
    return


def pil2array(pil):
    """
        - Purpose
            Convert a PIL image to a numpy array
        - Synopsis
            arr = pil2array(pil)
        - Input
            pil: The PIL image to convert.
        - Output
            arr: numpy array representing the PIL image.
        - Description
            Convert a PIL image to a numpy array. The array representing a
            RGB(A) image is formed by images stored sequencially: R-image,
            G-image, B-image and, optionally, Alpha-image.

    """

    import numpy
    w, h = pil.size
    binary = 0
    if pil.mode == '1':
        binary = True
        pil = pil.convert('L')
    if pil.mode == 'L':
        d = 1
        shape = (h,w)
    elif pil.mode == 'P':
        if 0:   # len(pil.palette.data) == 2*len(pil.palette.rawmode):
            binary = True
            pil = pil.convert('L')
            d = 1
            shape = (h,w)
        else:
            pil = pil.convert('RGB')
            d = 3
            shape = (h,w,d)
    elif pil.mode in ('RGB','YCbCr'):
        d = 3
        shape = (h,w,d)
    elif pil.mode in ('RGBA','CMYK'):
        d = 4
        shape = (h,w,d)
    else:
        raise TypeError, "Invalid or unimplemented PIL image mode '%s'" % pil.mode
    arr = numpy.reshape(numpy.fromstring(pil.tostring(), numpy.uint8, w*h*d), shape)
    if d > 1:
        arr = numpy.swapaxes(numpy.swapaxes(arr, 0, 2), 1, 2)
    if binary:
        return (arr > 0)
    return arr


def array2pil(arr):
    """
        - Purpose
            Convert a numpy array to a PIL image
        - Synopsis
            pil = array2pil(arr)
        - Input
            arr: numpy array to convert.
        - Output
            pil: The resulting PIL image.
        - Description
            Convert a numpy array to a PIL image. Use the conventions
            explained in the pil2array docstring.

    """
    import numpy
    nd = len(arr.shape)
    x = arr.astype('B')
    if nd == 2:
        d, h, w = (1,) + arr.shape
        mode = 'L'
    elif nd == 3:
        if arr.dtype == 'b':
            raise TypeError, "Binary array cannot be RGB"
        h, w, d = arr.shape
        if d == 3:   mode = 'RGB'
        elif d == 4: mode = 'RGBA'
        else:
            raise TypeError, "Array first dimension must be 1, 3 or 4 (%d)" % d
    else:
        raise TypeError, "Array must have 2 or 3 dimensions (%d)" % nd
    if d > 1:
        x = numpy.swapaxes(numpy.swapaxes(x, 1, 2), 0, 2)
    pil = Image.fromstring(mode, (w,h), x.tostring())
    if arr.dtype.char == '1':
        pil = pil.point(lambda i: i>0, '1')
    # return pil
    return pil

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
