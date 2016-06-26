"""
    Example run
    ```
        python3 preprocessing.py
    ```
"""
from PIL import Image
from scipy import ndimage
from skimage.filters import rank
from skimage.morphology import square
from skimage.morphology import disk 
from skimage.morphology import white_tophat
import numpy
import cv2 
import matplotlib.pyplot as plt
import PIL
from unbuffered import Unbuffered
import sys
# make print() not wait on the same buffer as the 
# def it exists in:
sys.stdout = Unbuffered(sys.stdout)
class Preprocess:
    """
        Preprocess class is responsible for anything preprocessing. It is build 
        for easy convolution of the preprocessing operations. Such that 
        operations may be easily followed by each other in any order by dotting 
        them out like so:
        ```
            obj =   Preprocess(
                        image="./STARE/im0255.ppm"
                    ).meanFilter(
                    ).show(
                    ).greyOpening(
                    ).show()
        ```
        Notice how `show()` can be called after any operation. `show()` uses the 
        PIL Image debugger to show the image.

        The implemented methods are generally limited to the methods describedin 
        Marin et al ITM 2011. However some methods allow for different 
        parameters to be used in the operation where the ones described in Marin 
        et al ITM 2011 are merely defaults.

        To run the methods described in Marin et al 2011 in the same order as 
        described then the method `process` can be used:
        ```
            obj =   Preprocess(
                        image="./STARE/im0003.ppm"
                    ).process(
                    ).show(
                    ).save(
                        path="./im0003_processed.png"
                    )
        ```

        Non standard requesites for running are:
            - scipy     https://www.scipy.org/
            - cv2       http://opencv-python-tutroals.readthedocs.io/en/latest/
            - skimage   http://scikit-image.org/

        @class Preprocess
        @param    image {string} The path to the image to be preprocessed.
        @param    maskTh {int} The threshold value to create the mask from
        @property source {string} Image source 
        @property image {PIL obj} PIL Image object
        @property mask  {numpy array} The mask matrix which is 0 in the area 
            outside FOV and 1's inside FOV
        @property threshold {int} The threshold value from which the mask is 
            made from. Lower intensity than threshold and the pixel is 
            considered outside FOV and inside otherwise. 
    """
    def __init__(self, image, maskTh=50):
        self.initialized = False
        self.__printStatus(
            "Initialize preprocessing for: " + image, 
            isEnd=True, 
            initial=True
        )
        self.source = image
        self.name = image.split("/")[-1].split(".")[0]
        self.image = Image.open(image)
        self.loaded = self.image.load()
        # self.threshold=50
        self.threshold = maskTh
        self.extractColorBands()
        self.mask = numpy.uint8(
            numpy.greater(
                self.red_array, 
                self.threshold
            ).astype(int)
        )

    def save(self, path, array=numpy.empty(0), useMask=False, rotate=True):
        """
            Saves the image array as png at the desired path. 
            
            @method save
            @param path {string} the path where the image will be saved.
            @param array {numpy array} The array which the image is made from, 
                default is self.image_array
            @param useMask {Bool} Wether to reset non FOV pixel using the mask. 
                Default is False 
        """
        if not array.any():
            array = self.image_array
        if useMask:
            array = array * self.mask
        self._arrayToImage(array).save(path, "png", rotate=rotate)
        self.__printStatus("saving to " + path + "...")
        self.__printStatus("[done]", True)
        return self

    def _arrayToImage(self, array=numpy.empty(0), rotate=True):
        """
            @private
            @method arrayToImage 
            @param array {numpy array} array which is converted to an image
            @param rotate {Bool} If true the image is transposed and rotated to 
                counter the numpy conversion of arrays.
        """
        self.__printStatus("array to image...")
        if not array.any():
            array = self.image_array
        img = Image.fromarray(numpy.uint8(array))
        self.__printStatus("[done]", True)
        if rotate:
            return img.transpose(Image.FLIP_TOP_BOTTOM).rotate(-90)
        else:
            return img 

    def show(
            self, 
            array=numpy.empty(0), 
            rotate=True, 
            invert=False, 
            useMask=False,
            mark=None
        ):
        """
            @method show
            @param array {numpy array} image array to be shown.
            @param rotate {Bool} Wether to rotate countering numpys array 
                conversion, default True.
            @param invert {Bool} Invert the image, default False.
            @param useMask {Bool} Reset non FOV pixels using the mask, default 
                is False.
        """
        if not array.any():
            array = self.image_array
        im = self._arrayToImage(array, rotate=rotate)
        self.__printStatus("show image...")
        if useMask:
            array = array * self.mask
        if mark:
            im = im.convert("RGB")
            pixels = im.load()
            x, y = mark
            for i in range(x-1, x+1):
                for j in range(y-1, y+1):
                    # color an area around the mark
                    # blue, for easilier visibility
                    pixels[i, j] = (0, 0, 255)
        if invert:
            Image.eval(im, lambda x:255-x).show()
        else:
            print("#####", im.mode, "#####")
            im.show()
        self.__printStatus("[done]", True)
        return self

    def extractColorBands(self):
        """
            Returns a greyscaled array from the green channel in 
            the original image. 

            @method extractColorBands
        """
        self.__printStatus("Extract color bands...")
        green_array = numpy.empty([self.image.size[0], self.image.size[1]], int)
        red_array = numpy.empty([self.image.size[0], self.image.size[1]], int)
        for x in range(self.image.size[0]):
            for y in range(self.image.size[1]):
                red_array[x,y] = self.loaded[x,y][0]
                green_array[x,y] = self.loaded[x,y][1]
        self.green_array = green_array
        self.red_array = red_array
        self.image_array = self.green_array
        self.__printStatus("[done]", True)
        return self

    def greyOpening(self, array=numpy.empty(0)):
        """
            Makes a 3x3 morphological grey opening 

            @method greyOpening 
            @param array {numpy array} array to operate on.
        """
        self.__printStatus("Grey opening...")
        if not array.any():
            array = self.image_array
        self.grey_opened = ndimage.morphology.grey_opening(array, [3,3])
        self.image_array = self.grey_opened * self.mask
        self.__printStatus("[done]", True)
        return self

    def meanFilter(self, m=3, array=numpy.empty(0)):
        """
            Mean filtering, replaces the intensity value, by the average 
            intensity of a pixels neighbours including itself. 
            m is the size of the filter, default is 3x3

            @method meanFilter
            @param m {int} The width and height of the m x m filtering matrix, 
                default is 3.
            @param array {numpy array} the array which the operation is carried 
                out on.
        """
        self.__printStatus("Mean filtering " + str(m) + "x" + str(m) + "...")
        if not array.any():
            array = self.image_array
        if array.dtype not in ["uint8", "uint16"]:
            array = numpy.uint8(array)
        mean3x3filter = rank.mean(array, square(m), mask=self.mask)
        self.image_array = mean3x3filter * self.mask
        self.__printStatus("[done]", True)
        return self

    def gaussianFilter(self, array=numpy.empty(0), sigma=1.8, m=9):
        """
            @method gaussianFilter
            @param array {numpy array} the array the operation is carried out 
                on, default is the image_array.
            @param sigma {Float} The value of sigma to be used with the gaussian 
                filter operation
            @param m {int} The size of the m x m matrix to filter with.
        """
        self.__printStatus(
            "Gaussian filter sigma=" + str(sigma) + ", m=" + str(m) + "..."
        )
        if not array.any():
            array = self.image_array
        self.image_array = cv2.GaussianBlur(array, (m,m), sigma) * self.mask
        self.__printStatus("[done]", True)
        return self

    def _getBackground(self, array=numpy.empty(0), threshold=None):
        """
            _getBackground returns an image unbiased at the edge of the FOV

            @method _getBackground
            @param array {numpy array} the array the operation is carried out 
                on, default is the image_array.
            @param threshold {int} Threshold that is used to compute a 
                background image, default is self.threshold. 
        """
        if not array.any():
            array = self.red_array
        if not threshold:
            threshold = self.threshold
        saved_image_array = self.image_array
        background = self.meanFilter(m=69).image_array
        self.__printStatus("Get background image...")
        # reset self.image_array
        self.image_array = saved_image_array
        for x in range(len(background)):
            for y in range(len(background[0])):
                if array[x,y] > threshold:
                    if x-35 > 0:
                        x_start = x-35
                    else:
                        x_start = 0
                    if x+34 < len(background):
                        x_end = x+34
                    else:
                        x_end = len(background) -1
                    if y-35 > 0:
                        y_start = y-35
                    else:
                        y_start = 0
                    if y+35 < len(background[0]):
                        y_end = y+35
                    else:
                        y_end = len(background[0]) -1
                    # 1 is added to the right and bottom boundary because of
                    # pythons way of indexing
                    x_end += 1
                    y_end += 1
                    # mask is is the same subMatrix but taken from the original 
                    # image array
                    mask    = array[x_start:x_end, y_start:y_end]                   
                    # indexes of the non fov images
                    nonFOVs = numpy.less(mask, threshold)
                    # indexes of FOVs 
                    FOVs    = numpy.greater(mask, threshold)
                    # subMat is a 69x69 matrix with x,y as center
                    subMat  = background[x_start:x_end, y_start:y_end]
                    # subMat must be a copy in order to not allocate values into 
                    # background directly
                    subMat = numpy.array(subMat, copy=True)
                    
                    subMat[nonFOVs] = subMat[FOVs].mean()
                    # finding every element less than 10 from the original image
                    # and using this as indices on the background subMatrix 
                    # is used to calculate the average from the 'remaining 
                    # pixels in the square' 
                    background[x,y] = subMat.mean()

        self.__printStatus("[done]", True)
        return background

    def subtractBackground(self, array=numpy.empty(0)):
        """
            @method subtractBackground 
            @param array {numpy array} the array the operation is carried out 
                on, default is the image_array.
        """ 
        if not array.any():
            array = self.image_array
        background = self._getBackground() * self.mask
        self.__printStatus("Subtract background...")
        self.image_array = numpy.subtract(
            numpy.int16(array), 
            numpy.int16(background)
        ) * self.mask
        self.__printStatus("[done]", True)
        return self

    def linearTransform(self, array=numpy.empty(0)):
        """
            Shade correction maps the background image into values 
            that fits the grayscale 8 bit images [0-255]
            from: http://stackoverflow.com/a/1969274/2853237

            @method linearTransform
            @param array {numpy array} the array the operation is carried out 
                on, default is the image_array.
        """
        self.__printStatus("Linear transforming...")
        if not array.any():
            array = self.image_array
        # Figure out how 'wide' each range is
        leftSpan = array.max() - array.min()
        rightSpan = 255
        array = ((array - array.min()) / leftSpan) * rightSpan
        self.image_array = array * self.mask
        self.__printStatus("[done]", True)
        return self

    def transformIntensity(self, array=numpy.empty(0)):
        """
            @method transformIntensity
            @param array {numpy array} the array the operation is carried out 
                on, default is the image_array. 
        """
        self.__printStatus("Scale intensity levels...")
        if not array.any():
            array = self.image_array
        
        counts = numpy.bincount(array.astype(int).flat)
        ginput_max = numpy.argmax(counts)
        for x in range(len(array)):
            for y in range(len(array[0])):
                st = str(array[x,y]) + " ==> "
                st += str(array[x,y] + 128 - ginput_max) + " "
                array[x,y] + 128 - ginput_max 
                if array[x,y] < 0:
                    array[x,y] = 0
                elif array[x,y] > 255:
                    array[x,y] = 255
        s = str(ginput_max)
        self.image_array = array * self.mask
        self.__printStatus("[done]", True)
        return self

    def vesselEnhance(self, array=numpy.empty(0)):
        """
            @method vesselEnhance
            @param array {numpy array} the array the operation is carried out
                on, default is the image_array.
        """
        self.__printStatus("Vessel enhancement...");
        if not array.any():
            array = self.image_array
        # disk shaped mask with radius 8 
        disk_shape = disk(8)
        # the complimentary image is saved to hc:
        array = numpy.uint8(array)
        hc = 255 - array
        # Top Hat transform
        # https://en.wikipedia.org/wiki/Top-hat_transform
        # White top hat is defined as the difference between
        # the opened image and the original image. 
        # in this case the starting image is the complimentary image `hc`
        self.image_array = white_tophat(hc, selem=disk_shape) * self.mask
        self.__printStatus("[done]", True)
        return self

    def __printStatus(self, status, isEnd=False, initial=False):
        """
            @private
            @method __printStatus
            @param status {string}
            @param isEnd {Bool} Wether to end with a newline or not, default is 
                false.
            @param initial {Bool} Wether this is the first status message to be 
                printed, default False.
        """
        if not initial and not isEnd:
            status = "\t" + status
        if initial:
            status = "\n" + status
        if isEnd:
            delim="\n"
        else:
            delim=""
            # set tabs so status length is 48
            tabs = ((48 - len(status)) // 8) * "\t"
            status += tabs
        print(status, end=delim, sep="")

    def process(self, enhance=True, onlyEnhance=False):
        """
            `process` starts the preprocess process described in 
            Marin et al ITM [2011]
            The article works with two types of preprocessed images. 
            The first is the convoluted image obtained with all operations 
            except for `vesselEnhance` denoted as a homogenized image. And the 
            second is the vessel enhanced image which is the convolution of the 
            vessel enhancement operation on the homogenized image. 

            This method supports both images. If `enhance` is False then 
            self.image_array will be of the homogenized image and afterwards the 
            vessel enhanced image can be computed without starting over by 
            setting `onlyEnhance` to True. So to compute both images one at a 
            time one could call:

            ```
                obj = Preprocess(
                        "./im0075.ppm"
                    )
                    .process(
                        enhance=False
                    ).show(
                    ).process(
                        onlyEnhance=True
                    ).show()
            ```

            @method process
            @method enhance {Bool} Wether to also process the vessel enhancement 
                operation or not, default True.
            @method onlyEnhance {Bool} Wether to only do the vessel enhancement 
                operation, default False.
        """
        if not onlyEnhance:
            self.greyOpening()
            self.meanFilter()
            self.gaussianFilter()
            self.subtractBackground()
            self.linearTransform()
            self.transformIntensity()
        if enhance or onlyEnhance:
            self.vesselEnhance()
        # returns the object where 
        # all described preprocess has taken place
        # available on self.feature_array or self.show(), self.save(<path>)
        return self