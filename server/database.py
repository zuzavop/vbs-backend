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


max_depth = 2


DATA = None
DATA_COLLECTIONS = {}
TIME_COLLECTIONS = {}
METADATA_COLLECTIONS = {}
LOCAL_DATA_COLLECTIONS = {}
CUR_SELECTION = None
TEXTURE_DATA_COLLECTIONS = {}
SPLITS_COLLECTIONS = {}
STR_IDS_COLLECTIONS = {}

# Class structure to store data, indices, labels, and the time
class memory_data_storage:
    DATA = None
    IDS = None
    LABELS = None
    TIME = None
    METADATA = None
    LOCAL_DATA = None
    TEXTURE_DATA = None

    def __init__(self, new_data=None, new_ids=None, new_labels=None, new_time=None, new_metadata=None, new_local_data=None, new_texture_data=None):
        self.DATA = new_data
        self.IDS = new_ids
        self.LABELS = new_labels
        self.TIME = new_time
        self.METADATA = new_metadata
        self.LOCAL_DATA= new_local_data
        self.TEXTURE_DATA= new_texture_data

def get_available_memory():
    mem_info = psutil.virtual_memory()
    available_memory = mem_info.available
    return available_memory


def get_data():
    return DATA.DATA


def get_ids():
    return DATA.IDS


def get_labels():
    if isinstance(DATA.LABELS, list):
        return torch.tensor([-1] * len(DATA.IDS))
    return DATA.LABELS


def get_splits():
    data_collection_name = f'{CUR_SELECTION[0]}-{CUR_SELECTION[1]}'
    return SPLITS_COLLECTIONS[data_collection_name]


def get_str_ids():
    data_collection_name = f'{CUR_SELECTION[0]}-{CUR_SELECTION[1]}'
    return STR_IDS_COLLECTIONS[data_collection_name]


def get_time(new_data=None):
    global DATA
    global TIME_COLLECTIONS

    if new_data is None and hasattr(DATA, 'TIME'):
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
    global DATA
    global METADATA_COLLECTIONS

    column_names = ['id', 'song name', 'album name', 'artist name', 'utc_time', 'local_time', 'latitude', 'longitude', 'semantic_name', 'time_zone', 'hour', 'weekday']

    if new_data is None and hasattr(DATA, 'METADATA'):
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


def get_local_data(new_data=None):
    global DATA
    global LOCAL_DATA_COLLECTIONS

    if new_data is None and hasattr(DATA, 'LOCAL_DATA'):
        type_local_file_name = DATA.LOCAL_DATA[next(iter(DATA.LOCAL_DATA))][:-13]
        l.logger.info(type_local_file_name)
        if type_local_file_name not in LOCAL_DATA_COLLECTIONS:
            LOCAL_DATA_COLLECTIONS[type_local_file_name] = {}
            for type_local, type_local_file in new_data.LOCAL_DATA.items():
                if type_local_file is not None:
                    with open(type_local_file, 'rb') as f:
                        tmp_data = dill.load(f)

                    if isinstance(tmp_data, list):
                        tmp_data = torch.tensor(np.array(tmp_data), dtype=torch.float32)
                    LOCAL_DATA_COLLECTIONS[type_local_file_name][type_local] = tmp_data
            else:
                return {}
        return LOCAL_DATA_COLLECTIONS[type_local_file_name]
    elif new_data is not None and hasattr(new_data, 'LOCAL_DATA'):
        if new_data.LOCAL_DATA == {} or new_data.LOCAL_DATA is None:
            return {}
        for type_local, type_local_file in new_data.LOCAL_DATA.items():
            type_local_file_name = type_local_file[:-13]
            if type_local_file_name not in LOCAL_DATA_COLLECTIONS:
                LOCAL_DATA_COLLECTIONS[type_local_file_name] = {}
                l.logger.info(type_local_file)
                if type_local_file is not None:
                    with open(type_local_file, 'rb') as f:
                        tmp_data = dill.load(f)

                    if isinstance(tmp_data, list):
                        tmp_data = torch.tensor(np.array(tmp_data), dtype=torch.float32)
                    LOCAL_DATA_COLLECTIONS[type_local_file_name][type_local] = tmp_data
            elif type_local_file_name in LOCAL_DATA_COLLECTIONS and type_local not in LOCAL_DATA_COLLECTIONS[type_local_file_name]:
                l.logger.info(type_local_file)
                if type_local_file is not None:
                    with open(type_local_file, 'rb') as f:
                        tmp_data = dill.load(f)

                    if isinstance(tmp_data, list):
                        tmp_data = torch.tensor(np.array(tmp_data), dtype=torch.float32)
                    LOCAL_DATA_COLLECTIONS[type_local_file_name][type_local] = tmp_data
    else:
        return {}


def name_splitter(dataset):
    if dataset in ['VBSLHE']:
        return '_'
    else:
        return '_'


def uri_spliter(id, dataset):
    if dataset in ['V3C', 'VBSLHE', 'LSC']:
        return id.split('_', 1)
    else:
        return id.rsplit('_', 1)


def set_data(new_data=None, new_ids=None, new_labels=None):
    global DATA
    DATA = memory_data_storage(new_data, new_ids, new_labels)


def load_features(dataset=c.BASE_DATASET, model=c.BASE_MODEL, first_load=False):
    global DATA
    global DATA_COLLECTIONS
    global CUR_SELECTION

    if dataset == '':
        dataset = c.BASE_DATASET
    if model == '':
        model = c.BASE_MODEL

    # Check if dataset and model are already loaded
    if [dataset, model] == CUR_SELECTION:
        return
    CUR_SELECTION = [dataset, model]

    data_collection_name = f'{dataset}-{model}'
    if data_collection_name in DATA_COLLECTIONS:
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
                get_local_data(DATA_COLLECTIONS[data_collection_name])
                get_texture_data(DATA_COLLECTIONS[data_collection_name])
                if cur_dataset == 'LSC':
                    get_metadata(DATA_COLLECTIONS[data_collection_name])

            # Convert data to torch.float32
            try:
                data_tensor = DATA_COLLECTIONS[data_collection_name].DATA
                DATA_COLLECTIONS[data_collection_name].DATA = data_tensor.to(torch.float32)

                separator = name_splitter(cur_dataset).encode()
                separator_ids = np.array([id.split(separator)[0] if cur_dataset == 'LSC' else id.rpartition(separator)[0] for id in DATA_COLLECTIONS[data_collection_name].IDS])
                splits = np.where(separator_ids[:-1] != separator_ids[1:])[0] + 1
                SPLITS_COLLECTIONS[data_collection_name] = np.r_[0, splits, len(separator_ids)]
                STR_IDS_COLLECTIONS[data_collection_name] = np.array([x.decode() if isinstance(x, bytes) else x for x in DATA_COLLECTIONS[data_collection_name].IDS])
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


def get_texture_data(new_data=None):
    global DATA
    global TEXTURE_DATA_COLLECTIONS

    if new_data is None and hasattr(DATA, 'TEXTURE_DATA'):
        type_local_file_name = DATA.TEXTURE_DATA[next(iter(DATA.TEXTURE_DATA))][:-13]
        if type_local_file_name not in TEXTURE_DATA_COLLECTIONS:
            TEXTURE_DATA_COLLECTIONS[type_local_file_name] = {}
            for type_local, type_local_file in new_data.TEXTURE_DATA.items():
                if type_local_file is not None:
                    with open(type_local_file, 'rb') as f:
                        tmp_data = dill.load(f)

                    if isinstance(tmp_data, list):
                        tmp_data = torch.tensor(np.array(tmp_data), dtype=torch.float32)
                    TEXTURE_DATA_COLLECTIONS[type_local_file_name][type_local] = tmp_data
            else:
                return {}
        return TEXTURE_DATA_COLLECTIONS[type_local_file_name]
    elif new_data is not None and hasattr(new_data, 'TEXTURE_DATA'):
        if new_data.TEXTURE_DATA == {} or new_data.TEXTURE_DATA is None:
            return {}
        for type_local, type_local_file in new_data.TEXTURE_DATA.items():
            type_local_file_name = type_local_file[:-13]
            if type_local_file_name not in TEXTURE_DATA_COLLECTIONS:
                TEXTURE_DATA_COLLECTIONS[type_local_file_name] = {}
                l.logger.info(type_local_file)
                if type_local_file is not None:
                    with open(type_local_file, 'rb') as f:
                        tmp_data = dill.load(f)

                    if isinstance(tmp_data, list):
                        tmp_data = torch.tensor(np.array(tmp_data), dtype=torch.float32)
                    TEXTURE_DATA_COLLECTIONS[type_local_file_name][type_local] = tmp_data
            elif type_local_file_name in TEXTURE_DATA_COLLECTIONS and type_local not in TEXTURE_DATA_COLLECTIONS[type_local_file_name]:
                l.logger.info(type_local_file)
                if type_local_file is not None:
                    with open(type_local_file, 'rb') as f:
                        tmp_data = dill.load(f)

                    if isinstance(tmp_data, list):
                        tmp_data = torch.tensor(np.array(tmp_data), dtype=torch.float32)
                    TEXTURE_DATA_COLLECTIONS[type_local_file_name][type_local] = tmp_data
    else:
        return {}

