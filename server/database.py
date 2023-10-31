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


# Class structure to store data and indices
class memory_data_storage:
    DATA = None
    IDS = None

    def __init__(self, new_data, new_ids):
        self.DATA = new_data
        self.IDS = new_ids


DATA = None
CUR_SELECTION = None


def get_data():
    return DATA.DATA


def get_ids():
    return DATA.IDS


def set_data(new_data, new_ids):
    global DATA
    DATA = memory_data_storage(new_data, new_ids)


def load_features(dataset=c.BASE_DATASET, model=c.BASE_MODEL):
    global DATA
    global CUR_SELECTION

    if [dataset, model] == CUR_SELECTION:
        return

    CUR_SELECTION = [dataset, model]

    l.logger.info('Start to load pre-generated embeddings')
    start_time = time.time()

    for dataset_and_feature in datasets_and_features:
        cur_dataset, cur_model, cur_file = dataset_and_feature
        if cur_dataset == dataset and cur_model == model:
            break

    # read data and ids from hard drive
    with open(cur_file, 'rb') as f:
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
