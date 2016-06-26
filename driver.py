"""
    Driver is the main driver class to run image preprocessing and feature 
    extraction with necessary image paths and helper classes etc. 
    
    Example run
    ```
        python3 driver.py
    ```
"""
import os, numpy, glob, re
from os import path
from PIL import Image
from preprocessing import Preprocess
from featureExtraction import FeatureExtraction
import json
from dictionary import text

"""
    The driver class is the entry point for initializing image preprocessing and
    subsequent feature extraction via the FeatureExtraction class and Preprocess 
    class. 

    The driver can be used interactively by just issuing 
    ```
        python3 main.py
    ```  
    or it can be included in another program by importing the driver class and 
    presetting required arguments via the configuration file.

    The default configuration file is `./configurations.json` where standard 
    JSON format is applied. The configuration file is also used by the driver 
    object to store the mean and standard deviation from the training set. This 
    way mean and standard deviation can be used to normalize the training set 
    and any future images.

    The configuration is seperated into two JSON objects where the first 
    `generalSettings` consists of user defined settings as follows:
        @param `DestinationFolder` {string} The folder path for saving the 
            computed features.
        @param `ImagePath` {string} The folder where the images are located
        @param `LabelPath` {string} The folder where the groundtruth images are 
            stored used as labels
        @param `extract_balanced` {bool} Wether to extract balanced feature sets 
            of 50/50 class 0 and class 1
        @param `override_params` {bool} If true, any stored mean and standard 
            deviation will be overwritten by the computed mean and standard 
            deviation from this set of extracted features.

    The other JSON object is the `paramaters` object which is used by the driver 
    object to read and write mean and standard deviation vectors to. 

    @class Driver
"""

class Driver:
    # initializers
    def __init__(
            self, 
            ImagePath=None, 
            LabelPath=None, 
            DestinationFolder=None, 
            mode=None
        ):
        self.configfile = "configurations.json"
        self.agree = ['y', "yes"]
        self.extract_balanced_notset = False
        self.override_params_notset  = False
        ImagePath, LabelPath, DestinationFolder = self._getConfiguration(
            ImagePath, 
            LabelPath, 
            DestinationFolder
        )
        if self.extract_balanced_notset:
            self._setExtractBalanced()
        if self._runTrainOrTestMode(mode):
            self._setImagePath(ImagePath)
            self._setLabelPath(LabelPath)
        
        else:
            self._setImagePath(ImagePath)
        self._setDestinationFolder(DestinationFolder)

    def _getConfiguration(self, 
        ImagePath=None, 
        LabelPath=None, 
        DestinationFolder=None):
        try:
            with open(self.configfile, 'r') as f:
                confs = json.load(f)
        except FileNotFoundError:
            self.extract_balanced_notset    = True
            self.override_params_notset     = True
            return [ImagePath, LabelPath, DestinationFolder]

        gsets = confs['generalSettings']
        if (gsets['ImagePath'] and not ImagePath):
            ImagePath = gsets['ImagePath']
        if (gsets['LabelPath'] and not LabelPath):
            LabelPath = gsets['LabelPath']
        if (gsets['DestinationFolder'] and not DestinationFolder):
            DestinationFolder = gsets['DestinationFolder']
        try:
            self.extract_balanced = gsets['extract_balanced']
        except KeyError: 
            self.extract_balanced_notset = True
        try:
            self.override_params  = gsets['override_params']  
        except KeyError:
            self.override_params_notset = True

        # returns ImagePath, LabelPath and DestinationFolder in order 
        # for any passed arguments to have precedence. And to reuse 
        # the same path validation functions.
        return [ImagePath, LabelPath, DestinationFolder]
    

    def _runTrainOrTestMode(self, mode):
        if mode:
            self.mode = mode
            return mode == "training"
        print(text['train_or_test'])

        if input(text['choose_mode']).lower() in self.agree:
            self.mode = "training"
            return True
        else: 
            self.mode = "testing"
            return False

    def _setDestinationFolder(self, path=None, insist=True):
        destPath = path or input(text['choose_dest_folder'])
        if os.path.exists(destPath):
            self.destinationPath = destPath
        else:
            print(text["dest_folder"], destPath, text["no_path"])
            if (insist):
                self._setDestinationFolder() 

    def _setImagePath(self, path=None, insist=True):
        ImagePath = path or input(text['type_image_path'])
        if os.path.exists(ImagePath):
            self.ImagePath = ImagePath
        else:
            print(text['image_folder'], text["no_path"])
            if (insist):
                self._setImagePath()

    def _setLabelPath(self, path=None, insist=True):
        LabelPath = path or input(text['type_label_path'])
        if os.path.exists(LabelPath):
            self.LabelPath = LabelPath
        else:
            print(text["label_folder"], text["no_path"])
            if (insist):
                self._setLabelPath()

    def _setExtractBalanced(self):
        self.extract_balanced = input(text['type_ex_bal']) in self.agree

    def _getStoredMeanStd(self, confs, override=False):
        if override:
            confs["parameters"]["featureMeans"] = self.mean.tolist()
            confs["parameters"]["featureStd"]   = self.std.tolist()
            with open(self.configfile, 'w') as f:
                json.dump(
                    confs, 
                    f, 
                    sort_keys=True, 
                    indent=4, 
                    separators=(',', ': ')
                )

        mean    = numpy.array(confs["parameters"]["featureMeans"])
        std     = numpy.array(confs["parameters"]["featureStd"])
        return [mean, std]       


    """
        @method run
        @param `extract_balanced` if set to `True` the extracted feature set
        will include an equal amount of nonvessel and vessel pixels. 
    """
    def run(self):
        print(text["run_mode"].format(self.mode))
        if self.mode == "training":
            labels = [f for f in os.listdir(self.LabelPath) 
                if path.isfile(path.join(self.LabelPath, f))]
            if not len(labels):
                raise FileNotFoundError(
                    2,
                    "No labels found in labelpath", 
                    self.LabelPath
                )

            images = []
            for label in labels:
                name = re.search("^im[0-9]{4}", label)
                img = name.group(0) + ".ppm"
                images.append(img)
            if len(labels) == len(images) or not len(labels):
                print(text["disp_label_dir"], self.LabelPath)
                print(text["disp_image_dir"], self.ImagePath)
                print(text["disp_dest_dir"], self.destinationPath)
                print(text["disp_conf_file"], self.configfile)
                meanVectors = numpy.empty((len(labels), 7))
                varVectors  = numpy.empty((len(labels), 7))
                feObj = []
                for i, x in enumerate(images):
                    print("\nImage ", i+1, " of ", len(images))
                    fe = FeatureExtraction(
                        image=path.join(self.ImagePath, images[i]), 
                        GTPath=path.join(self.LabelPath, labels[i])
                    )
                    fe.computeFeatures().normalize(comp_only=True)
                    # save mean and standard deviation from training:
                    feObj.append(fe)
                    meanVectors[i]  = fe.mean_vector
                    varVectors[i]   = fe.var_vector
                self.mean    = meanVectors.mean(axis=0)
                self.std     = numpy.sqrt(varVectors.mean(axis=0))

                try:
                    with open(self.configfile, 'r') as f:
                        confs = json.load(f)
                except FileNotFoundError:
                    print("Creating configfile ", self.configfile)
                    with open(self.configfile, "w+") as f:
                        # make sure params are written, since the
                        #  list is empty.
                        self.override_params_notset = False
                        self.override_params = True
                        self._setExtractBalanced()
                        confs = {
                                "generalSettings": {
                                    "DestinationFolder": "",
                                    "ImagePath": "",
                                    "LabelPath": ""                                },
                                "parameters": {
                                    "featureMeans": [
                                    ],
                                    "featureStd": [
                                    ]
                                }
                            }                    
                if self.override_params_notset: 
                    if input(text["write_mean_std"]).lower() in self.agree:
                        self.mean, self.std = self._getStoredMeanStd(
                            confs, 
                            True
                        )
                    else:
                        self.mean, self.std = self._getStoredMeanStd(confs)
                elif self.override_params:
                    self.mean, self.std = self._getStoredMeanStd(
                        confs, 
                        True
                    )
                else: 
                    self.mean, self.std = self._getStoredMeanStd(confs)

                ## Normalize data with mean of all training image features
                ## and write features to csv files 
                for i, fe in enumerate(feObj):
                    print(
                        "Normalizing ", 
                        fe.source, 
                        " and writing features to ", 
                        images[i] + ".csv"
                    )
                    fe.normalize(
                        mean=self.mean, 
                        std=self.std
                    ).exportCSV(
                        filename=path.join(
                            self.destinationPath, 
                            images[i] + ".csv"
                        ), 
                        balanced=self.extract_balanced
                    )
            else:
                print("Some label names are not as exptected")
                print("Names are expected to start with '^im[0-9]{4}'")
                print("e.g, im0001.ah.ppm")
        else:
            images = [f for f in os.listdir(self.ImagePath) 
                if path.isfile(path.join(self.ImagePath, f))]
            print(text["disp_image_dir"], self.ImagePath)
            print(text["disp_dest_dir"], self.destinationPath)
            print(text["disp_conf_file"], self.configfile)
            with open(self.configfile, 'r') as f:
                confs = json.load(f)
            if (confs["parameters"]["featureMeans"] and 
                confs["parameters"]["featureStd"]):

                mean    =  numpy.array(confs["parameters"]["featureMeans"])
                std     =  numpy.array(confs["parameters"]["featureStd"])

            else:

                print(
                    "No mean and standard deviation exists in ", 
                    self.configfile
                )
                return

            for i, x in enumerate(images): 
                print("\nImage ", i+1, " of ", len(images))
                fe = FeatureExtraction(
                    image=path.join(
                        self.ImagePath, 
                        images[i]
                    )
                )
                fe.computeFeatures().normalize(
                    mean=mean, 
                    std=std
                ).exportCSV(
                    filename=path.join(self.destinationPath, images[i] + ".csv")
                )
