import os
import dill
import h5py
import time

import pandas as pd
import numpy as np

from tqdm import tqdm

import logger as l
import configs as c


# Selector for the database loader
features_to_use = 'features-laion'
if c.MODEL == 'laion':
    features_to_use = 'features-laion'


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
    root_directory = os.path.join(c.DATABASE_ROOT, features_to_use)

    # Internal storage
    internal_storage = os.path.join(
        c.DATABASE_PROCESSED, f'pickled_files_{c.MODEL}.pkl'
    )
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

        # Now, file_paths contains all the file paths in the directory structure
        # Loop two times through the file paths to reduce memory
        tmp_data = []
        for file_path in tqdm(file_paths):
            if 'hdf5' in file_path:
                h5_file = h5py.File(file_path, 'r')
                tmp_data.append(np.array(h5_file['data']))

                del h5_file

        # Concatenate list of data to one data array
        data = np.concatenate(tmp_data, axis=0)
        del tmp_data

        tmp_ids = []
        for file_path in tqdm(file_paths):
            if 'hdf5' in file_path:
                h5_file = h5py.File(file_path, 'r')
                tmp_ids.append(np.array(h5_file['ids']))

                del h5_file

        # Concatenate list of ids to one ids array
        ids = np.concatenate(tmp_ids, axis=0)
        del tmp_ids

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


def load_msb(video_id, frame_id):
    l.logger.info('Start to load MSB')

    # Specify the directory
    tsv_file_path = os.path.join(c.DATABASE_ROOT, 'msb', f'{video_id}.tsv')
    data = pd.read_csv(tsv_file_path, sep='\t')

    return data[data['startframe'] == frame_id]
