"""
    A class that takes the prediction array from the classifier and
    turns it back into an image.
"""
import numpy, math
from PIL import Image

d = None
t = None
class PredictedImage:

    def __init__(self, data=None):
        if not data:
            raise ValueError
        else:
            self.next = 0
            self.data=data
            data = numpy.array(data)
            lesser = numpy.less(data, 0)
            greater = numpy.greater(data, 1)
            data[greater] = 255
            data[lesser] = 0
            global d
            d = data
            imageLen = 700*605
            array = numpy.array(data, dtype='uint8')
            global t
            t = array
            self.image_arrays = []  
            if len(array) > imageLen:
                if len(array) % imageLen:
                    raise ValueError
                for img in range(int(len(array)/ imageLen)):
                    self.image_arrays.append(array[img*imageLen:(img+1) * imageLen])
            elif len(array) == imageLen:
                self.image_arrays.append(array)
            else:
                raise ValueError
            self.images = [] 
            for img in self.image_arrays:
                image = img.reshape((700, 605))
                self.images.append(Image.fromarray(image))


    def showNext(self, index=None):
        print("index", index, "index")
        if not index:
            index = self.next
        # print(index, len(self.images))
        print("index", index, "index")
        if (index < len(self.images)):
            self.images[index].show()
            self.next += 1
        else:
            print("No more images to show")
    def showAll(self):
        if len(self.images):
            for img in self.images:
                img.show()
        else:
            print("There are no images to show")


    def showOverlay(self, labelImage, index=None):
        if not index:
            index = self.next
        labelImage = Image.open(labelImage).convert("RGB")
        # labelImage = labelImage.convert("RGB")
        img = self.image_arrays[index]
        img = img.reshape(700, 605)
        lbl = numpy.array(labelImage)
        lbl[numpy.greater(img.transpose(), 0)] = (255, 0, 0)
        labelImage = Image.fromarray(lbl)
        # img[numpy.greater(img, 0)] = 
        # img = self.image_arrays[index][self.images_arrays[index].greater()] 
        labelImage.show()
        self.image = labelImage
        return self

    def save(self, path):
        if self.image:
            self.image.save(path)
        return self 

