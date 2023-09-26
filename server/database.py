import os
import h5py

import numpy as np

from tqdm import tqdm

import logger as l

import dill

import time


# Class structure to store data and indices
class memory_data_storage:
    DATA = None
    IDS = None

    def __init__(self, new_data, new_ids):
        self.DATA = new_data
        self.IDS = new_ids


DATA = None


def get_data():
    return DATA.DATA


def get_ids():
    return DATA.IDS


def set_data(new_data, new_ids):
    global DATA
    DATA = memory_data_storage(new_data, new_ids)


def normalize(data):
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    return data


def load_features():
    global DATA
    l.logger.info('Start to load pre-generated embeddings')

    # Specify the root directory you want to start walking from
    root_directory = '/data/vbs/'

    # Internal storage
    internal_storage = '/data/vbs/processed/pickled_files.pkl'
    if not os.path.exists(internal_storage):
        start_time = time.time()

        # Initialize an empty list to store file paths
        file_paths = []

        # Walk through the directory structure
        for folder, subfolders, files in os.walk(root_directory):
            for file in files:
                # Join the folder and file name to create the full file path
                file_path = os.path.join(folder, file)
                # Append the file path to the list
                file_paths.append(file_path)

        ids = None
        data = None
        # Now, file_paths contains all the file paths in the directory structure
        for file_path in tqdm(file_paths):
            if 'hdf5' in file_path:
                h5_file = h5py.File(file_path, 'r')

                if data is None:
                    data = np.array(h5_file['data'])
                else:
                    data = np.concatenate((data, np.array(h5_file['data'])), axis=0)

                # get video id and key frame
                if ids is None:
                    ids = np.array(h5_file['ids'])
                else:
                    ids = np.concatenate((ids, np.array(h5_file['ids'])), axis=0)

                del h5_file

        del file_paths

        # normalize to reduce numbers
        data = normalize(data)

        # store data and ids in memory
        set_data(data, ids)

        del data
        del ids

        # store data and ids on hard drive
        os.makedirs(os.path.dirname(internal_storage), exist_ok=True)
        with open(internal_storage, 'wb+') as f:
            dill.dump(DATA, f)

        execution_time = time.time() - start_time
        l.logger.info(
            f'Reading in and converting features took: {execution_time:.6f} secs'
        )

    else:
        start_time = time.time()

        # read data and ids from hard drive
        with open(internal_storage, 'rb') as f:
            DATA = dill.load(f)

        execution_time = time.time() - start_time
        l.logger.info(f'Reading in features took: {execution_time:.6f} secs')

    l.logger.info(get_data().shape)
    l.logger.info('Finished to load pre-generated embeddings')
