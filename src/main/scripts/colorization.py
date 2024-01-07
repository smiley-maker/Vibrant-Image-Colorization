from src.main.lib.models.CNN import CNN
from src.main.lib.dataHandlers.pokemonHandler import pokemonHandler
import numpy as np
import os

if __name__ == "__main__":    
    # sets up a pokemonHandler object
    handler = pokemonHandler()
    handler.collectData()

    print(f"training data has {len(handler.xTrain)} samples")

    # Creating a CNN object
    cnn = CNN(0.001)

    cnn.model.fit(handler.xTrain, handler.yTrain, batch_size=handler.BATCH_SIZE)