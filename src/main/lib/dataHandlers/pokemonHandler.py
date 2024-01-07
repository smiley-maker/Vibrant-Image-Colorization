from src.main.utils.data_storage import read_hdf5, store_hdf5
from keras.preprocessing.image import ImageDataGenerator
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import logging
import cv2
import os


class pokemonHandler():
    DATA_PATH = "src/main/data/pokemon/"

    IMG_SIZE = (256, 256)
    BATCH_SIZE = 32

    TRAINING_DATA_NAME = "train_data.h5"
    TEST_DATA_NAME = "test_data.h5"

    def __init__(self):
        """

        This class manages the flow of data with the option for either batch-wise
        data gathering and processing for model training or to obtain the x and y
        data all at once. Additionally, functions are provided for saving and reading
        .h5 files containing data. 

        """

        # Variables to store training and testing data
        self.xTrain = None
        self.xTest = None
        self.yTrain = None
        self.yTest = None

        # Ensuring data is available
        if os.path.exists(pokemonHandler.DATA_PATH + "POKEMON/"):
            # Get a list of files (excluding directories) in the directory
            files_in_directory = [f for f in os.listdir(pokemonHandler.DATA_PATH + "POKEMON/") if os.path.isfile(os.path.join(pokemonHandler.DATA_PATH + "POKEMON/", f))]

            if not files_in_directory:
                # Downloads the data if there are no files in the data directory. 
                self.downloadRawData()
            else:
                print(f"Data available at {pokemonHandler.DATA_PATH + 'POKEMON/'}, beginning training.")


        # Generators for training and testing data
#        self.datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
#        self.train_generator = self.setup_generator(subset='training')
#        self.test_generator = self.setup_generator(subset='validation')

    

    def downloadRawData(self):
        """
        Obtains data from Kaggle using the API. Only run once (or if you need to
        redownload for some reason).
        """
        api = KaggleApi()
        api.authenticate()

        dataset_name = "dollarakshay/pokemon-images"

        if not os.path.exists(pokemonHandler.DATA_PATH):
            os.mkdir(pokemonHandler.DATA_PATH)

        api.dataset_download_files(dataset_name, path=pokemonHandler.DATA_PATH, unzip=True)
    
    
    def setup_generator(self, subset):

        if not os.path.exists(pokemonHandler.DATA_PATH + subset + "/"):
            os.makedirs(pokemonHandler.DATA_PATH + subset)

        return self.datagen.flow_from_directory(
            pokemonHandler.DATA_PATH + "raw_images/",
            target_size=self.IMG_SIZE,
            batch_size=pokemonHandler.BATCH_SIZE,
            class_mode=None,  # Set to 'input' for X, 'none' for Y
            subset=subset
        )

    
    def preprocessBatch(self, batch):
        """Preprocesses a batch of images for colorization, handling single images and batches.

        Args:
            batch: A NumPy array of images with shape (batch_size, height, width, channels)
                or a single image with shape (height, width, channels).

        Returns:
            A tuple of preprocessed X and Y data, both as NumPy arrays with shapes
            (batch_size, height, width, channels).
        """

        if len(batch) < self.BATCH_SIZE:  # Single image
            try:
                print(len(batch))
                img = cv2.cvtColor(batch, cv2.COLOR_BGR2RGB)

                # Reassess necessity of grayscale conversion
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                img = img.astype(np.float32)
                img = cv2.resize(img, (256, 256))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

                X_batch = img[:, :, 0:1][np.newaxis, ...]  # Add a batch dimension
                Y_batch = img[:, :, 1:][np.newaxis, ...]  # Add a batch dimension

            except Exception as e:
                logging.error(f"An issue occurred trying to preprocess the image:\n{e}")

        else:  # Batch of images
            X_batch = np.empty((batch.shape[0], pokemonHandler.IMG_SIZE[0], pokemonHandler.IMG_SIZE[1], 1), dtype=np.float32)
            Y_batch = np.empty((batch.shape[0], pokemonHandler.IMG_SIZE[0], pokemonHandler.IMG_SIZE[1], 2), dtype=np.float32)

            for i, input_img in enumerate(batch):
                try:
                    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

                    # Reassess necessity of grayscale conversion
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                    img = img.astype(np.float32)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

                    X_batch[i] = img[:, :, 0:1]
                    Y_batch[i] = img[:, :, 1:]

                except Exception as e:
                    logging.error(f"An issue occurred trying to preprocess image {i + 1}:\n{e}")

        print(f"After preprocessing, X_batch has the following shape: {X_batch.shape}")
        print(f"After preprocessing, Y_batch has the following shape: {Y_batch.shape}")

        return X_batch, Y_batch


    

    def collectData(self):
        """
        This function offers a way of collecting all of the data in an image directory. 
        It uses image generators to get the data and preprocess it, and sets the xTrain,
        yTrain, xTest, and yTest class variables appropriately. 

        Returns:
            pokemonHandler: returns self to allow nested function calling
                            (e.g. collectData().preprocessBatch(...).saveData()...)
        """

        # Define parameters for data augmentation and preprocessing
        testSplit = 0.2

        # ImageDataGenerator for data augmentation and normalization
        datagen = ImageDataGenerator(rescale=1./255, validation_split=testSplit)

        # Load and split the dataset into training and testing sets
        train_generator = datagen.flow_from_directory(
            directory=pokemonHandler.DATA_PATH + "raw_images/",
            target_size=pokemonHandler.IMG_SIZE,
            class_mode='input',  # Use 'input' for autoencoders
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            directory=pokemonHandler.DATA_PATH + "raw_images/",
            target_size=pokemonHandler.IMG_SIZE,
            class_mode='input',
            subset='validation'
        )

        print(len(train_generator))

        # Loads and preprocesses training data
        self.xTrain = self.get_all_data(train_generator)

        # Loads and preprocesses testing data
        self.xTest = self.get_all_data(validation_generator)        

        return self
    

    def get_all_data(self, generator):
        """
        This function steps through all of the batches in a generator object
        and returns the data as a list. 

        Args:
            generator (ImageDataGenerator): Takes in a generator for either the
                                            training or testing data

        Returns:
            Numpy Array: Returns an array containing all of the data obtained by
                         the batch loading process. 
        """
        data_list = [] # List to store data

        for _ in range(len(generator)):
            batch_data, _ = next(generator) # Gets batch from the input generator
            x, y = self.preprocessBatch(batch_data)                
            data_list.append((x,y)) # Appends that data to list
        
        # Returns the data as a single list, rather than multiple batch lists
        return data_list
   

    def returnData(self):
        """
        Returns the class variables for training and testing data as tuples

        Returns:
            Tuple: Returns the current training and testing class data
        """
        return (self.xTrain, self.yTrain), (self.xTest, self.yTest)
        
    
    def saveData(self, toSave : str = "train") :
        """
        Saves the specified data to h5 files

        Args:
            toSave (str, optional): A string indicating whether the train or test data is being saved.
                                    Either "train" or "test" 
                                    Defaults to "train".

        Raises:
            FileNotFoundError: If the directory where the data is intended to be saved is not found

        Returns:
            pokemonHandler: returns self to allow nested function calling
                            (e.g. collectData().preprocessBatch(...).saveData()...)
        """
        if not os.path.exists(pokemonHandler.DATA_PATH):
            raise FileNotFoundError(f"Directory Not Found at [{pokemonHandler.DATA_PATH }]")
                
        # Checks which data to save and ensures that the data exists
        if toSave == "train" and self.xTrain is not None and self.yTrain is not None:
            try:
                # Stores training data as h5 file at the specified location
                store_hdf5(self.xTrain, self.yTrain, pokemonHandler.DATA_PATH + "saved_data/", pokemonHandler.TRAINING_DATA_NAME)
            except: 
                print("Something went wrong while trying to save training data...")
                print(f"x train has a length of {len(self.xTrain)}; y train has a length of {len(self.yTrain)}")
        
        elif toSave == "test" and self.xTest is not None and self.yTest is not None:
            try:
                # Stores testing data as h5 file at the specified location
                store_hdf5(self.xTest, self.yTest, pokemonHandler.DATA_PATH + "saved_data/", pokemonHandler.TEST_DATA_NAME)
            except: 
                print("Something went wrong while trying to save testing data...")
                print(f"x test has a length of {len(self.xTest)}; y test has a length of {len(self.yTest)}")
        
        elif toSave != "train" and toSave != "test": 
            logging.warning("toSave must be either train or test to save any data")

        else :
            logging.warning(f"{toSave} data hasn't been initialized, try running collectData before this function.")
        
        return self

    def readLocalData(self) :
        """
        Reads h5 files from the directories specified above as class constants. 

        Returns:
        pokemonHandler: returns self to allow nested function calling
                        (e.g. collectData().preprocessBatch(...).saveData()...)
        """

        try: 
            # Reads h5 file for training data and sets class variables
            self.xTrain, self.yTrain = read_hdf5(pokemonHandler.DATA_PATH + "saved_data/", pokemonHandler.TRAINING_DATA_NAME)
        except FileNotFoundError as e:
            print("the file wasn't found, have you generated & saved the training data into an h5 file yet? ")
        except: 
            print("an unknown error occured...")
        
        try: 
            # Reads h5 file for testing data and sets class variables 
            self.xTest, self.yTest = read_hdf5(pokemonHandler.DATA_PATH + "saved_data/", pokemonHandler.TEST_DATA_NAME)
        except FileNotFoundError as e:
            print("the test data can't be found, have you run collectData yet? ")
        except: 
            print("an unknown error occured...")

        return self

if __name__ == "__main__": 
    handler = pokemonHandler().collectData().processData().saveData().readLocalData()