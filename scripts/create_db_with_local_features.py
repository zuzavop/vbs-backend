# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse

import os
import dill
import h5py
import time

import torch

import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import datetime


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
        self.LOCAL_DATA = new_local_data
        self.TEXTURE_DATA = new_texture_data


# Normalization
def normalize(features):
    return features / features.norm(dim=-1, keepdim=True)


def extend_with_local_features(db_dir, model):
    # Get all database files (ending with .pkl) excluding those with 'db_time'. 'db_metadata' and 'db_local' in the name
    pkl_files = [
        file
        for file in os.listdir(os.path.join(os.path.dirname(db_dir), 'processed'))
        if file.endswith('.pkl') and 'db_time' not in file and 'db_metadata' not in file and 'local' not in file and 'texture' not in file and model in file
    ]

    # Initialize an empty list to store file paths
    file_paths = []

    # Walk through the directory structure
    for folder, subfolders, files in os.walk(db_dir):
        for file in files:
            if file.endswith('.hdf5'):
                # Join the folder and file name to create the full file path
                file_path = os.path.join(folder, file)
                # Append the file path to the list
                file_paths.append(file_path)

    loaded_file_path = {}
    type_local_data = db_dir.split("/")[-1].split("_")[0]

    for file in pkl_files:
        print(f'Adding to database file: {file}')
        internal_storage = os.path.join(os.path.join(os.path.dirname(db_dir), "processed"), file)

        try:
            print(f'Opening file {internal_storage}')
            with open(internal_storage, 'rb') as f:
                data = dill.load(f)

            # Create a separate file for local data based on the original database file
            localdata_storage = f'{internal_storage[:-4]}_{type_local_data}_local.pkl'

            # Check conditions for local data existence or validity in the data
            if (
                not os.path.exists(localdata_storage)
                or data.LOCAL_DATA is None
                or data.LOCAL_DATA == ''
                or not hasattr(data, 'LOCAL_DATA')
                or type_local_data not in data.LOCAL_DATA
            ):
                features = []

                print(f'Starting to add local features to the database')

                # Extract video IDs and frame IDs from the database
                ids = np.array(data.IDS).astype(str)

                # Iterate over unique IDs, extracting corresponding local data from hdf5 files
                for id in tqdm(ids):
                    splits = id.split('-', 1)
                    if len(splits) < 2:
                        splits = id.rsplit('_', 1)
                    video_id, frame_id = splits
                    local_data_file = os.path.join(os.path.join(db_dir, video_id),video_id + "-" + model + ".hdf5")

                    if local_data_file in file_paths:
                        try:
                            if local_data_file in loaded_file_path:
                                hdf5_file = loaded_file_path[local_data_file]
                            else:
                                hdf5_file = h5py.File(local_data_file, 'r')
                                loaded_file_path[local_data_file] = hdf5_file

                            id_byte = id.encode('utf-8')
                            feature_index = np.where(np.array(hdf5_file['ids']) == id_byte)[0][0]

                            features.append(torch.from_numpy(hdf5_file['data'][feature_index]))
                        except Exception as e:
                            print(f'Could not load {id} with {e}')

                if isinstance(features, list):
                    features = torch.tensor(np.array(features), dtype=torch.float32)
                else:
                    features = features.to(torch.float32)

                print(f'Found: {len(features)}, Ids: {len(ids)}')

                # Update the original data with the new timestamp storage path
                if not isinstance(data.LOCAL_DATA, dict):
                    data.LOCAL_DATA = {}
                data.LOCAL_DATA[type_local_data] = localdata_storage

                # Write the updated data back to the original database file
                with open(internal_storage, 'wb') as f:
                    dill.dump(data, f)
                print(f'Successful writing file {internal_storage}')

                # Write the extracted local data to the newly created local data storage file
                with open(localdata_storage, 'wb') as f:
                    dill.dump(features, f)
                print(f'Successful writing file {localdata_storage}')

                # Cleanup: delete temporary variables
                del features
                del data

            else:
                # If local data are already present, inform the user
                print(f'Already contains local features {internal_storage}')

        except Exception as e:
            # Handle exceptions during file processing
            print(f'Failed with file {internal_storage} and {e}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to extend all databases with local features')
    parser.add_argument('--db-dir', type=str, help='Directory containing the database files')
    parser.add_argument('--model-name', type=str, help='Model name')

    args = parser.parse_args()

    db_dir = args.db_dir
    model = args.model_name

    extend_with_local_features(db_dir, model)
