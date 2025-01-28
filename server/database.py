# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import dill
import h5py
import time
import torch
import psutil

import pandas as pd
import numpy as np

from tqdm import tqdm

import logger as l
import configs as c


# max depth to search for 'processed' folders
max_depth = 2


# global variables
# memory_data_storage object to store the current used dataset
DATA = None

# dict to store all datasets and their features - key: dataset name with model name, value: memory_data_storage object
DATA_COLLECTIONS = {}

# dict to store timestamps for each dataset - key: path to pickle file, value: timestamps for each image from the dataset (timestamp of the video frame that the image represents) from the pickle file
TIME_COLLECTIONS = {}

# dict to store metadata information for eahc dataset - key: path to pickle file, value: metadata for each image from the dataset (contains additional information like place, weekday, etc.) from the pickle file
METADATA_COLLECTIONS = {}

# list to store the current selected dataset and model
CUR_SELECTION = None

# dict to store the splits of the current selected dataset and model - key: dataset name with model name, value: list of indices where the dataset is split - split dataset to each videos
SPLITS_COLLECTIONS = {}


class memory_data_storage:
    """ Class to store data of each dataset in memory
    
    Attributes:
        DATA: torch.tensor - feature vectors of images from the dataset
        IDS: list - global ids of images from the dataset (ususally the filename)
        LABELS: torch.tensor - index of labels of images to the nounlist (if available) from the dataset
        TIME: str - path to the pickle file with timestamps of the dataset used as key into the TIME_COLLECTIONS
        METADATA: str - path to the pickle file with metadata of the dataset used as key into the METADATA_COLLECTIONS
    """
    DATA = None
    IDS = None
    LABELS = None
    TIME = None
    METADATA = None

    def __init__(self, new_data=None, new_ids=None, new_labels=None, new_time=None, new_metadata=None):
        self.DATA = new_data
        self.IDS = new_ids
        self.LABELS = new_labels
        self.TIME = new_time
        self.METADATA = new_metadata


def get_available_memory():
    """ Get available memory of the system"""
    mem_info = psutil.virtual_memory()
    available_memory = mem_info.available
    return available_memory


def get_data():
    return DATA.DATA


def get_ids():
    return DATA.IDS


def get_labels():
    """ Get labels of the dataset if available otherwise return list with -1"""
    if isinstance(DATA.LABELS, list):
        return torch.tensor([-1] * len(DATA.IDS))
    return DATA.LABELS


def get_splits():
    """ Get splits of the dataset if available otherwise return empty array"""
    data_collection_name = f'{CUR_SELECTION[0]}-{CUR_SELECTION[1]}'
    return SPLITS_COLLECTIONS[data_collection_name]


def get_time(new_data=None):
    """ Get timestamps of the dataset if available otherwise return empty array"""
    global DATA
    global TIME_COLLECTIONS

    if new_data is None and hasattr(DATA, 'TIME'):
        l.logger.info(DATA.TIME)
        if DATA.TIME not in TIME_COLLECTIONS:
            if DATA.TIME is not None:
                with open(DATA.TIME, 'rb') as f:
                    tmp_data = dill.load(f)
                if DATA.TIME not in TIME_COLLECTIONS:
                    TIME_COLLECTIONS[DATA.TIME] = np.array(tmp_data)
            else:
                return np.array([])
        return TIME_COLLECTIONS[DATA.TIME]
    elif new_data is not None and hasattr(new_data, 'TIME'):
        if new_data.TIME not in TIME_COLLECTIONS:
            l.logger.info(new_data.TIME)
            if new_data.TIME is not None:
                with open(new_data.TIME, 'rb') as f:
                    tmp_data = dill.load(f)
                TIME_COLLECTIONS[new_data.TIME] = np.array(tmp_data)
    else:
        return np.array([])
    
    
def get_metadata(new_data=None):
    """ Get metadata of the dataset if available otherwise return empty dataframe
    Metadata contains additional information like place, weekday, etc. """
    global DATA
    global METADATA_COLLECTIONS
    
    column_names = ['id', 'song name', 'album name', 'artist name', 'utc_time', 'local_time', 'latitude', 'longitude', 'semantic_name', 'time_zone', 'hour', 'weekday']
    
    if new_data is None and hasattr(DATA, 'METADATA'):
        l.logger.info(DATA.METADATA)
        if DATA.METADATA not in METADATA_COLLECTIONS:
            if DATA.METADATA is not None:
                tmp_data = pd.read_pickle(DATA.METADATA)
                tmp_data = pd.DataFrame(tmp_data, columns=column_names)
                METADATA_COLLECTIONS[DATA.METADATA] = tmp_data
            else:
                return pd.DataFrame()
        return METADATA_COLLECTIONS[DATA.METADATA]
    elif new_data is not None and hasattr(new_data, 'METADATA'):
        if new_data.METADATA not in METADATA_COLLECTIONS:
            l.logger.info(new_data.METADATA)
            if new_data.METADATA is not None:
                tmp_data = pd.read_pickle(new_data.METADATA)
                tmp_data = pd.DataFrame(tmp_data, columns=column_names)
                METADATA_COLLECTIONS[new_data.METADATA] = tmp_data
    else:
        return pd.DataFrame()


def name_splitter(dataset='V3C'):
    """ Get the separator used in the ids of the dataset"""
    return '_'


def uri_spliter(id, dataset):
    """ Split the id of the image to get the video id and the image id"""
    if dataset in ['V3C', 'VBSLHE', 'LSC']:
        return id.split('_', 1)
    else:
        return id.rsplit('_', 1)


def set_data(new_data=None, new_ids=None, new_labels=None):
    """ Set new data, ids and labels to the global variables"""
    global DATA
    DATA = memory_data_storage(new_data, new_ids, new_labels)


def load_features(dataset=c.BASE_DATASET, model=c.BASE_MODEL, first_load=False):
    """ Load pre-generated embeddings from hard drive
    
    Args:
        dataset (str): Name of the dataset
        model (str): Name of the model
        first_load (bool): If True, load only the first available model
    """
    global SPLITS_COLLECTIONS
    global DATA
    global DATA_COLLECTIONS
    global CUR_SELECTION

    l.logger.info(f'Selected dataset: {dataset}, model: {model}')

    if dataset == '':
        dataset = c.BASE_DATASET
    if model == '':
        model = c.BASE_MODEL

    # Check if dataset and model are already loaded
    if [dataset, model] == CUR_SELECTION:
        l.logger.info(f'Already selected dataset: {dataset}, model: {model}')
        return
    CUR_SELECTION = [dataset, model]

    data_collection_name = f'{dataset}-{model}'
    if data_collection_name in DATA_COLLECTIONS:
        l.logger.info(f'Already loaded dataset: {dataset}, model: {model}')
        DATA = DATA_COLLECTIONS[data_collection_name]
        return

    l.logger.info('Start to load pre-generated embeddings')
    start_time = time.time()

    datasets_and_features = []
    # Use os.walk to search for 'processed' folders up to two levels deep
    for root, _, files in os.walk(c.DATABASE_ROOT):
        current_depth = root.count(os.path.sep) - c.DATABASE_ROOT.count(os.path.sep)
        if current_depth <= max_depth:
            if 'processed' in root:
                for file in files:
                    if file.endswith('.db.pkl'):
                        file_path = os.path.join(root, file)
                        file_name = os.path.basename(file_path)
                        dataset_name = file_path.split('/')[-3]
                        feature_name = file_name.split('__')[-1].split('.')[0]
                        datasets_and_features.append(
                            [dataset_name, feature_name, file_path]
                        )

    # Get sizes of all datasets
    full_sizes = 0
    available_mem = get_available_memory()
    l.logger.info(f'Available memory: {available_mem}')

    # Check available models and datasets
    file_path = None
    for dataset_and_feature in datasets_and_features:
        cur_dataset, cur_model, cur_file = dataset_and_feature
        l.logger.info(
            f'Found dataset: {cur_dataset}, model: {cur_model}, file: {cur_file}'
        )

        full_sizes += os.path.getsize(cur_file)
        data_collection_name = f'{cur_dataset}-{cur_model}'
        if full_sizes < available_mem and data_collection_name not in DATA_COLLECTIONS:
            # read data and ids from hard drive
            with open(cur_file, 'rb') as f:
                DATA_COLLECTIONS[data_collection_name] = dill.load(f)
                get_time(DATA_COLLECTIONS[data_collection_name])
                get_metadata(DATA_COLLECTIONS[data_collection_name])
                
            # Convert data to torch.float32
            try:
                data_tensor = DATA_COLLECTIONS[data_collection_name].DATA
                DATA_COLLECTIONS[data_collection_name].DATA = data_tensor.to(torch.float32)
                
                separator = name_splitter(cur_dataset).encode()
                separator_ids = np.array([id.split(separator)[0] if cur_dataset == 'LSC' else id.rpartition(separator)[0] for id in DATA_COLLECTIONS[data_collection_name].IDS])
                splits = np.where(separator_ids[:-1] != separator_ids[1:])[0] + 1
                SPLITS_COLLECTIONS[data_collection_name] = np.r_[0, splits, len(separator_ids)]
            except Exception as e:
                l.logger.error(f"Error converting data to torch.float32: {e}")
                raise

        if cur_dataset == dataset and cur_model == model:
            file_path = cur_file
            if not first_load:
                break
        elif cur_dataset == dataset:
            file_path = cur_file

    if file_path:
        data_collection_name = f'{dataset}-{model}'
        if data_collection_name in DATA_COLLECTIONS:
            DATA = DATA_COLLECTIONS[data_collection_name]
        else:
            # read data and ids from hard drive
            with open(file_path, 'rb') as f:
                DATA = dill.load(f)

        execution_time = time.time() - start_time
        l.logger.info(f'Reading in features took: {execution_time:.6f} secs')
        l.logger.info(f'Got size: {list(get_data().shape)}')
        l.logger.info('Finished to load pre-generated embeddings')
    else:
        l.logger.error('File to load pre-generated embeddings not found:')
        l.logger.error(f'{file_path}')
