from PIL import Image
import subprocess
import numpy
import os, re
from os import path

class VisualizeFeatures:

    def __init__ (self, csvPath=None):
        self.features       = numpy.genfromtxt(csvPath, delimiter=",")
        self.name           = csvPath.split("/")[-1].split(".")[0]
        print("Showing image: ", self.name)
        for f in range(self.features.shape[1]):
            # if f != 2:
            #     continue
            self.transformed = self.transformGrayScale(self.features[:,f])
            self.show(self.transformed, f)
            input("Press enter to continue...")
    
    def transformGrayScale(self, array=numpy.empty(0)):
        if not array.any():
            raise ValueError
        maxVal = array.max() 
        minVal = array.min()
        leftSpan = maxVal - minVal
        rightSpan = 255
        array = ((array - minVal) / leftSpan) * rightSpan
        return array

    def save(self, pth, array=numpy.empty(0)):
        if not array.any():
            array = self.transformed
        img = array.reshape((700, 605)).transpose()   
        img = Image.fromarray(img)
        if img.mode == "F":
            img = img.convert("RGB")
        print("Saving to: ", path.join(pth, self.name + ".png"))
        img.save(path.join(pth, self.name + ".png"))

    def show(self, array=numpy.empty(0), feature=None):
        if not array.any():
            array = self.transformed
        img = array.reshape((700, 605)).transpose()   
        img = Image.fromarray(img)
        img.show()
        img = img.convert("RGB")
