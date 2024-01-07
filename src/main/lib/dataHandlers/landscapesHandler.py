from keras.utils import to_categorical
import numpy as np
import os
import logging
import cv2
from kaggle.api.kaggle_api_extended import KaggleApi

from src.main.utils.data_storage import read_hdf5, store_hdf5
#from src.main.utils.data_storage import store_hdf5, read_hdf5

class landscapeHandler():
    NUM_CLASSES = 10

    DATA_PATH = "src/main/data/"

    DATA_PATH_RAW = "landscapes12k/raw/"
    DATA_PATH_PROCESSED = "landscapes12/processed/"

    TRAINING_DATA_NAME = "trainingData.h5"
    TEST_DATA_NAME = "testData.h5"

    def __init__(self):
        self.xTrainRaw = None
        self.yTrainRaw = None
        self.xTestRaw = None
        self.yTestRaw = None

        self.xTrain = None
        self.xTest = None
        self.yTrain = None
        self.yTest = None


    def collectData(self):
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files('utkarshsaxenadn/landscape-recognition-image-dataset-12k-images', 'BiT-LR-91-83.h5')


#        (self.xTrainRaw, self.yTrainRaw), (self.xTestRaw, self.yTestRaw) = mnist.load_data()

        return self
    
    def processData(self):
        self.xTrain = self.xTrainRaw.astype('float32') / 255.0
        self.xTrain = self.xTrain.reshape(self.xTrain.shape + (1,))
        self.xTest = self.xTestRaw.astype('float32') / 255.0
        self.xTest = self.xTest.reshape(self.xTest.shape + (1,))

        self.yTrain = self.yTrainRaw
        self.yTest = self.yTestRaw

        return self
    
    def clearRawData(self):
        self.xTrainRaw = None
        self.yTrainRaw = None
        self.xTestRaw = None
        self.yTestRaw = None

        return self

    def clearProcessedData(self):
        self.xTrain = None
        self.xTest = None
        self.yTrain = None
        self.yTest = None

        return self
    
    def getProcessedData(self):
        return (self.xTrain, self.yTrain), (self.xTest, self.yTest)
    
    def getRawData(self):
        return (self.xTrainRaw, self.yTrainRaw), (self.xTestRaw, self.yTestRaw)
    
    def saveData(self) :
        if not os.path.exists(landscapeHandler.DATA_PATH):
            raise FileNotFoundError(f"Directory Not Found at [{landscapeHandler.DATA_PATH}]")
        
        filePathRaw = landscapeHandler.DATA_PATH + landscapeHandler.DATA_PATH_RAW
        filePathPROCESSED = landscapeHandler.DATA_PATH + landscapeHandler.DATA_PATH_PROCESSED
        
        if self.xTrainRaw is not None and self.yTrainRaw is not None and self.xTestRaw is not None and self.yTestRaw is not None :
            if not os.path.exists(filePathRaw):
                os.makedirs(filePathRaw)
            store_hdf5(self.xTrainRaw, self.yTrainRaw, filePathRaw, landscapeHandler.TRAINING_DATA_NAME)
            store_hdf5(self.xTestRaw, self.yTestRaw, filePathRaw, landscapeHandler.TEST_DATA_NAME)
        else :
            logging.warning(f"Raw Data is None Cannot Save Raw Data")

        if self.xTrain is not None and self.yTrain is not None and self.xTest is not None and self.yTest is not None :
            if not os.path.exists(filePathPROCESSED):
                os.makedirs(filePathPROCESSED)
            store_hdf5(self.xTrain, self.yTrain, filePathPROCESSED, landscapeHandler.TRAINING_DATA_NAME)
            store_hdf5(self.xTest, self.yTest, filePathPROCESSED, landscapeHandler.TEST_DATA_NAME)
        else :
            logging.warning(f"Processed Data is None Cannot Save Processed Data")
        
        return self

    def readLocalData(self) :
        filePathRaw = landscapeHandler.DATA_PATH + landscapeHandler.DATA_PATH_RAW
        filePathPROCESSED = landscapeHandler.DATA_PATH + landscapeHandler.DATA_PATH_PROCESSED

        if os.path.isfile(filePathRaw+landscapeHandler.TRAINING_DATA_NAME) and os.path.isfile(filePathRaw+landscapeHandler.TEST_DATA_NAME):
            self.xTrainRaw, self.yTrainRaw = read_hdf5(filePathRaw, landscapeHandler.TRAINING_DATA_NAME)
            self.xTestRaw, self.yTestRaw = read_hdf5(filePathRaw, landscapeHandler.TEST_DATA_NAME)
        else :
            logging.warning(f"File [{landscapeHandler.TRAINING_DATA_NAME} or {landscapeHandler.TEST_DATA_NAME}] Not Found at [{filePathRaw}] Cannot Load Raw Data")

        if os.path.isfile(filePathPROCESSED+landscapeHandler.TRAINING_DATA_NAME) and os.path.isfile(filePathPROCESSED+landscapeHandler.TEST_DATA_NAME):
            self.xTrain, self.yTrain = read_hdf5(filePathPROCESSED, landscapeHandler.TRAINING_DATA_NAME)
            self.xTest, self.yTest = read_hdf5(filePathPROCESSED, landscapeHandler.TEST_DATA_NAME)
        else :
            logging.warning(f"File [{landscapeHandler.TRAINING_DATA_NAME} or {landscapeHandler.TEST_DATA_NAME}] Not Found at [{filePathPROCESSED}] Cannot Load Processed Data")
        
        return self

if __name__ == "__main__": 
    handler = landscapeHandler().collectData().read_hdf5().processData().saveData().readLocalData()