### Feature extraction

This repository includes python classes to extract a set of 7 features from 
retinal images and save them as `.csv` files, as described in Marin et al. (2011). 

The feature extraction can be done for each retinal image in a folder via the 
`Driver` class in **driver.py**.
The path to the images can be given by the `ImagePath` parameter. For extracting 
a training set, the path to label or groundtruth images can be given `LabelPath` 
parameter. The path to a folder where the extracted features are saved to as 
`.csv` files is given by `DestinationFolder`. 

A `mode` parameter can be set to `"training"` or `"testing"` to indicate whether
a training or test set is wanted.

So to extract a training set from images in folder named *STARE/* using labels in 
*labels/* and saving the features to *features/* the following could be issued:

```
    from driver import Driver
    driverObj = Driver(
                    ImagePath           =   "./STARE/", 
                    LabelPath           =   "./labels/",
                    DestinationFolder   =   "./features/",
                    mode                =   "training"
                )
```


The features can be used with any classifier by importing the resulting `.csv` files.

### Visualizing predictions
A **predictedImage.py** provides a way of visualizing the predicted images on top of 
the original image or the groundtruth image. 

Assuming the predictions are available as a flat python list called data, the following
could be used to make such images:
```
    from predictedImage import PredictedImage
    predImg = PredictedImage(data)
    predImg.showOverlay(labelImage="./labels/someImage.ppm").save(path="./predictedImage.png")
```

Notice that `PredictedImage` assumes that the flat data array represents predictions for 
each pixel in an image with a 700x605 format, which is used by the images in the STARE
database. 

### Debug features

The features saved in a `.csv` file can be visualized using the `VisualizeFeatures` class
in **visualizeFeatures.py**. 

Example:

```
    from visualizeFeatures import VisualizeFeatures
    visFeat = VisualizeFeatures(csvPath="./someFeatures.csv")
```

The visualization will show each feature at a time using the image debugger in the 
Python Image Library (PIL).
