import numpy as np
import h5py

def store_hdf5(data, meta, filePath, fileName):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        data        data tuple to be stored
        meta        meta data to be stored
    """
    # Create a new HDF5 file
    file = h5py.File(f"{filePath + fileName}", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "data", np.shape(data), h5py.h5t.STD_U8BE, data=data
    )
    meta_set = file.create_dataset(
        "meta", np.shape(meta), h5py.h5t.STD_U8BE, data=meta
    )
    file.close()


def read_hdf5(filePath, fileName):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        data        Data tuples retrieved
        meta      associated meta data, int label
    """
    data, meta = [], []

    # Open the HDF5 file
    file = h5py.File(f"{filePath + fileName}", "r+")

    data = np.array(file["/data"]).astype("uint8")
    meta = np.array(file["/meta"]).astype("uint8")

    return data, meta