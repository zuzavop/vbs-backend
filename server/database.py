import os
import h5py

import numpy as np

from tqdm import tqdm

import logger as l

import dill


# Class structure to store data and indices
class memory_data_storage:
    DATA = None
    IDX = None

    def __init__(self, new_data, new_idx):
        self.DATA = new_data
        self.IDX = new_idx


DATA = None


def get_data():
    return DATA.DATA


def get_idx():
    return DATA.IDX


def set_data(new_data, new_idx):
    global DATA
    DATA = memory_data_storage(new_data, new_idx)


def load_features():
    l.logger.info('Start to load pre-generated embeddings')

    # Specify the root directory you want to start walking from
    root_directory = '/data/vbs/'

    # Internal storage
    internal_storage = '/data/internal/internal.pkl'
    if not os.path.exists(internal_storage):
        # Initialize an empty list to store file paths
        file_paths = []

        # Walk through the directory structure
        for folder, subfolders, files in os.walk(root_directory):
            for file in files:
                # Join the folder and file name to create the full file path
                file_path = os.path.join(folder, file)
                # Append the file path to the list
                file_paths.append(file_path)

        tmp_ids = []
        tmp_data = []
        # Now, file_paths contains all the file paths in the directory structure
        for file_path in tqdm(file_paths):
            if 'hdf5' in file_path:
                h5_file = h5py.File(file_path, 'r')
                data = np.array(h5_file['data'])
                tmp_data.append(data)

                # get video id and key frame
                ids = np.array(h5_file['ids'])
                tmp_ids.append(ids)

        # Stack data after collecting it
        data = np.concatenate(tmp_data, axis=0)
        ids = np.concatenate(tmp_ids, axis=0)

        set_data(data, ids)

        os.makedirs(os.path.dirname(internal_storage), exist_ok=True)
        with open(internal_storage, 'wb+') as f:
            dill.dump(DATA, f)

    else:
        with open(internal_storage, 'rb') as f:
            data = dill.load(f)
            set_data(data.DATA, data.IDX)

    l.logger.info(get_data().shape)
    l.logger.info('Finished to load pre-generated embeddings')
