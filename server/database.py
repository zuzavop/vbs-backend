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


def get_available_memory():
    mem_info = psutil.virtual_memory()
    available_memory = mem_info.available
    return available_memory


# Class structure to store data, indices, and labels
class memory_data_storage:
    DATA = None
    IDS = None
    LABELS = None

    def __init__(self, new_data=None, new_ids=None, new_labels=None):
        self.DATA = new_data
        self.IDS = new_ids
        self.LABELS = new_labels


DATA = None
DATA_COLLECTIONS = {}
CUR_SELECTION = None


def get_data():
    return DATA.DATA


def get_ids():
    return DATA.IDS


def get_labels():
    if isinstance(DATA.LABELS, list):
        return torch.tensor([-1] * len(DATA.IDS))
    return DATA.LABELS


def name_splitter(ids, dataset):
    if dataset in ['MVK', 'VBSLHE']:
        return ids.split('-', 1)
    else:
        return ids.split('_', 1)


def set_data(new_data=None, new_ids=None, new_labels=None):
    global DATA
    DATA = memory_data_storage(new_data, new_ids, new_labels)


def load_features(dataset=c.BASE_DATASET, model=c.BASE_MODEL):
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
    l.logger.info(available_mem)

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

        if cur_dataset == dataset and cur_model == model:
            file_path = cur_file

    l.logger.info(f'Selected dataset: {dataset}, model: {model}')
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
