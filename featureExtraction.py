import scipy
import numpy
from PIL import Image
from preprocessing import Preprocess 
import cv2
from unbuffered import Unbuffered
import sys
from skimage.morphology import binary_erosion
from skimage.morphology import square
# make print() not wait on the same buffer as the 
# def it exists in:
sys.stdout = Unbuffered(sys.stdout)

class FeatureExtraction:
    """
        @param image {string} Source of raw image
        @param images {list} Source of preprocessed images. Where images[0] is 
        the homogenized image and images[1] is the vessel enhanced image.
        @param GTPath {string} Wether image has ground truth. If the image has 
        ground truth image with correct labels then gt is a path to the 
        groundtruth image.
    """
    def __init__(self, image=False, images=[], GTPath=""):
        if images:
            try:
                self.homogenized = numpy.array(Image.open(images[0]))
                self.vesselEnhanced = numpy.array(Image.open(images[1]))
                self.images = images
            except IndexError:
                print("""`images` parameter must include the homogenized image 
                    at `images[0]` and vessel enhanced image at `images[1]`""")
                raise
        else:
            self.preprocess          = Preprocess(image)
            self.homogenized    = self.preprocess.process(enhance=False).image_array
            self.vesselEnhanced = self.preprocess.process(onlyEnhance=True).image_array
            self.mask           = self.preprocess.mask
            self.source         = image
            self.image          = Image.open(image)
            self.loaded         = self.image.load()
        if len(GTPath):
            self.gt             = True
            self.groundtruth    = Image.open(GTPath)
        else:
            self.gt             = False

        self.feature_array      = numpy.empty(0)

    def __getHomogenized(self, forceNew=False):
        raise NotImplementedError
    """
        `exportCSV` exports `self.feature_array` to `filename` unless `array` 
        parameter is set if `balanced` then the exported features will have an 
        equal amount of class 0 and class 1. 
        The parameter `delim` can be used to change the seperator from commas to
        some other character.

        @method exportCSV
        @param filename {string} Name of file including the path to it where 
            features will be exported to
        @param array {numpy array} The feature array to export
        @param delim {string} The delimeter 
        @default ","
        @param balanced {bool} Wether to export the full feature array or a 
            balanced version with equal class representation
        @default False
    """
    def exportCSV(
            self, 
            filename="", 
            array=numpy.empty(0), 
            delim=",", 
            balanced=False
        ):

        if not array.any():
            array = self.feature_array
        if balanced:
            zeros   = array[numpy.less(array[:,0], 1)] 
            ones    = array[numpy.greater(array[:,0], 0)]
            if len(zeros) > len(ones):
                indices = numpy.random.choice(
                    len(zeros), 
                    size=len(ones), 
                    replace=False
                )
                ones = numpy.concatenate(
                    (ones, zeros[indices]), 
                    axis=0
                )
                array = ones
            if len(ones) > len(zeros):
                indices = numpy.random.choice(
                    len(ones), 
                    size=len(zeros), 
                    replace=False
                )
                zeros = numpy.concatenate(
                    (zeros, ones[indices]), 
                    axis=0
                )
                array = zeros
        if not len(filename):
            if hasattr(self, "source"):
                filename = "extracted_" + self.source
            else:
                filename = "extracted_" + self.images[1]
        if self.gt:
            formatting = ['%d', '%.0f', '%.0f', '%f', '%f', '%.0f', '%f', '%f']
            header = """label,\tfeat. 1,\tfeat. 2,\tfeat. 3,\tfeat. 4,\tfeat. 5,
                     \tHu mom. 1,\tHu mom. 2"""
        else:
            formatting = ['%.0f', '%.0f', '%f', '%f', '%.0f', '%f', '%f']
            header = """feat. 1,\tfeat. 2,\tfeat. 3,\tfeat. 4,\tfeat. 5,
                     \tHu mom. 1,\tHu mom. 2"""
        numpy.savetxt(
            filename,               
            array,                  
            fmt=formatting,         # formatting
            delimiter=',\t',        # column delimiter
            newline='\n',           # new line character
            footer='end of file',   # file footer
            comments='# ',          # character to use for comments
            header=header)          # file header
    """
        `normalize` is used to normalize the feature_array. If comp_only 
        (compute only) is set to `True` then only `self.std_vector` and 
        `self.mean_vector` will be set but the value of `self.feature_array` 
        will not be set. This can be useful if computing an accumulated mean and 
        standard deviation and then using the `mean` and `std` parameter later 
        to normalize with the accumulated mean and average standard deviation 
        vectors. 

        @method normalize 
        @param array {numpy array} The feature array if not set then 
            `self.feature_array`.
        @param mean {numpy array} The mean to use in the normalization. If not 
            set then it will be computed over the inside FOV pixels of the 
            `array` using the `self.mask`. 
        @param std {numpy array} The standard deviation to be used in 
            normalization. 
        @param comp_only {bool} If true then mean, sample variance and standard 
            deviation will be computed and saved to `self.var_vector`, 
            `self.std_vector` and `self.mean_vector` respectively. But they wont
            be used to normalize the feature array. 
        @default False
    """
    def normalize(
            self, 
            array=numpy.empty(0), 
            mean=numpy.empty(0), 
            std=numpy.empty(0), 
            comp_only=False
        ):
        if not array.any():
            array = self.feature_array
        # preserve label column
        # compute mean and std excluding out of FOV pixels
        indices = numpy.greater(self.mask.flatten(), 0)
        FOV = array[indices]

        # Since mean should only be computed on the training set
        # the assumption of ignoring the first column is made, since
        # this is the label column.
        if not mean.any():
            mean    = FOV.mean(axis=0)[1:]
        if not std.any():
            std     = FOV.std(axis=0)[1:]
            var     = FOV.var(axis=0)[1:]
        if comp_only:
            self.var_vector     = var
            self.std_vector     = std
            self.mean_vector    = mean
        else:
            if self.gt:
                labels = array[:,0]
                array[:,1:] = (array[:,1:] - mean) / std
            else:
                array = (array - mean) / std
            if self.gt:
                array[:,0] = labels
                # since there is a groundtruth then the first column
                # will be the label column, the rest are the actual features.d
            self.feature_array = array
        return self

    def computeFeatures(self, forceNew=False):
        if forceNew:
            return self._extract()

        elif self.feature_array.any():
            return self
        else:
            return self._extract()

    """
        `_extract` is responsible of extracting the feature array for every 
        pixel in the preprocessed image. If optional parameters 
        `homogenized_array` and `ve_array` are not provided then  

        @method _extract 
        @param homogenized_array {numpy array} The homogenized image from 
            preprocessing
        @param ve_array {numpy array} The vessel enhanced image from 
            preprocessing
    """
    def _extract(
            self, 
            homogenized_array=numpy.empty(0), 
            ve_array=numpy.empty(0)
        ):
        if not homogenized_array.any():
            homogenized_array = self.homogenized
        if not ve_array.any():
            ve_array = self.vesselEnhanced
        # erode image using an eroded mask 
        mask = binary_erosion(self.mask, square(10)) 
        homogenized_array = homogenized_array * mask
        # # # # # # # # # # # # # # # # # # # # #
        print("Extracting features ", end=" ")
        print("\t\t[", end="")
        self.feature_array = []
        for x in range(len(homogenized_array)):
            for y in range(len(homogenized_array[0])):
                if self.mask[x,y] or True: # disabled for now
                    #########################################
                    xstart  = x - 8 if x-8 >= 0 else 0
                    ystart  = y - 8 if y-8 >= 0 else 0

                    xend    = x + 8 if x+8 < len(ve_array) else len(ve_array) -1
                    yend    = y + 8 if y+8 < len(ve_array[0]) else len(ve_array[0]) -1
                    # 1 is added to the right and bottom boundary because of
                    # pythons way of indexing
                    xend += 1
                    yend += 1
                    
                    subarea = ve_array[xstart:xend, ystart:yend]

                    if subarea.max() != 0:
                    

                        Hu0, Hu1 = self.__moments(subarea)

                        ########################################
                        xstart  = x-4 if x-4 >= 0 else 0
                        ystart  = y-4 if y-4 >= 0 else 0

                        xend    = (x+4 
                            if x+4 < len(homogenized_array) 
                            else len(homogenized_array) -1)
                        yend    = (y+4 
                            if y+4 < len(homogenized_array[0]) 
                            else len(homogenized_array[0]) -1)
                        # 1 is added to the right and bottom boundary because of
                        # pythons way of indexing
                        xend += 1
                        yend += 1

                        subarea = homogenized_array[xstart:xend, ystart:yend]
                        FOV     = numpy.greater(subarea, 0)
                        subarea = (subarea[FOV] 
                            if FOV.any() and homogenized_array[x,y] > 0 
                            else numpy.array([0]))
                        # equation 5 from Marin et al.
                        f1      = homogenized_array[x,y] - subarea.min()
                        # equation 6 from Marin et al.
                        f2      = subarea.max() - homogenized_array[x,y]
                        # equation 7 from Marin et al.
                        f3      = homogenized_array[x,y] - subarea.mean()
                        # equation 8 from Marin et al.
                        f4      = subarea.std()
                        # equation 9 from Marin et al. 
                        # inverting the background, so setting zero to 255
                        f5      = homogenized_array[x,y]
                        ########################################

                        if self.gt:
                            # values in groundtruth are either 255 or 0
                            gtval = self.groundtruth.getpixel((x,y))
                            label = gtval if gtval == 0 else 1
                            features = [label, f1, f2, f3, f4, f5, Hu0, Hu1]
                        else:
                            features = [f1, f2, f3, f4, f5, Hu0, Hu1]

                    elif not self.gt:
                        features = [0, 0, 0.0, 0.0, 0, 0.0, 0.0]

                    else:
                        # values in groundtruth are either 255 or 0
                        gtval = self.groundtruth.getpixel((x,y))
                        label = gtval if gtval == 0 else 1
                        features = [label, 0, 0, 0.0, 0.0, 0, 0.0, 0.0]

                    self.feature_array.append(features)
            if x % (len(homogenized_array) * 0.05) < 1: 
                print("#", end="")

        self.feature_array = numpy.array(self.feature_array)
        print("]")
        return self

    """
        `__moments` computes the first two Hu moment over some array given by 
        the parameter `subarray`. 

        @private
        @method __moments
        @param subarray {numpy array} The area which the Hu moments are computed 
            over. 
    """
    def __moments(self, subarray):
        """
            I_HU(x,y) = subarray(x,y) * gaussian_matrix(x,y)

            returns absolute value of the log of the first two Hu moments
        """
        I_HU = self.__gausMatrix(subarray)
        h1, h2 = cv2.HuMoments(cv2.moments(I_HU))[0:2]
        h1 = numpy.log(h1) if h1 != 0 else h1
        h2 = numpy.log(h2) if h2 != 0 else h2
        return numpy.absolute( [h1[0], h2[0]] )

    def __gausMatrix(self, array, mu=0.0, sigma=1.7):
        x, y = array.shape
        return scipy.ndimage.filters.gaussian_filter(array, 1.7)