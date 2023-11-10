import os
import dill
import h5py
import time

import pandas as pd
import numpy as np

from tqdm import tqdm

import logger as l
import configs as c


max_depth = 2


db_pkl_files = []
# Use os.walk to search for 'processed' folders up to two levels deep
for root, _, files in os.walk(c.DATABASE_ROOT):
    current_depth = root.count(os.path.sep) - c.DATABASE_ROOT.count(os.path.sep)
    if current_depth <= max_depth:
        if 'processed' in root:
            for file in files:
                if file.endswith('.db.pkl'):
                    db_pkl_files.append(os.path.join(root, file))

datasets_and_features = []
for file in db_pkl_files:
    file_name = os.path.basename(file)
    dataset_name = file.split('/')[-3]
    feature_name = file_name.split('__')[-1].split('.')[0]
    datasets_and_features.append([dataset_name, feature_name, file])


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
CUR_SELECTION = None


def get_data():
    return DATA.DATA


def get_ids():
    return DATA.IDS


def get_labels():
    return DATA.LABELS


def set_data(new_data=None, new_ids=None, new_labels=None):
    global DATA
    DATA = memory_data_storage(new_data, new_ids, new_labels)


def load_features(dataset=c.BASE_DATASET, model=c.BASE_MODEL):
    global DATA
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

    # Check available models and datasets
    file_path = None
    for dataset_and_feature in datasets_and_features:
        cur_dataset, cur_model, cur_file = dataset_and_feature
        if file == dataset and cur_model == model:
            file_path = cur_file
            break

    if file_path:
        # read data and ids from hard drive
        with open(file_path, 'rb') as f:
            DATA = dill.load(f)

        execution_time = time.time() - start_time
        l.logger.info(f'Reading in features took: {execution_time:.6f} secs')
        l.logger.info(get_data().shape)
        l.logger.info('Finished to load pre-generated embeddings')
    else:
        l.logger.error('File to load pre-generated embeddings not found:')
        l.logger.error(f'{file_path}')


def load_msb(video_id, frame_id):
    l.logger.info('Start to load MSB')

    # Specify the directory
    tsv_file_path = os.path.join(c.DATABASE_ROOT, 'msb', f'{video_id}.tsv')
    data = pd.read_csv(tsv_file_path, sep='\t')

    return data[data['startframe'] == frame_id]
