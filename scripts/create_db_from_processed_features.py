# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse

import os
import dill
import h5py
import time

import torch

import numpy as np

from tqdm import tqdm


# Class structure to store data, indices, and labels
class memory_data_storage:
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


# Normalization
def normalize(features):
    return features / features.norm(dim=-1, keepdim=True)


def load_and_save_features(DATABASE_ROOT, MODEL):
    # Specify the root directory you want to start walking from
    root_directory = os.path.dirname(DATABASE_ROOT)

    # Get last directory for the dataset name
    last_directory = os.path.basename(DATABASE_ROOT)
    print(f'Start with {MODEL} for {last_directory} on {root_directory}')

    # Internal storage
    internal_storage = os.path.join(
        root_directory, f'processed/{last_directory}__{MODEL}.db.pkl'
    )
    print(f'Internal Storage at: {internal_storage}')

    # If the file does not exist, create new datbase file
    if not os.path.exists(internal_storage):
        start_time = time.time()

        # Initialize an empty list to store file paths
        file_paths = []

        # Walk through the directory structure
        for folder, subfolders, files in os.walk(DATABASE_ROOT):
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
                # tmp_data.append(np.array(h5_file['data']))
                tmp_data.append(torch.from_numpy(h5_file['data'][:]))

                del h5_file

        # Concatenate list of data to one data array
        # data = np.concatenate(tmp_data, axis=0)
        data = torch.cat(tmp_data)
        del tmp_data

        tmp_ids = []
        for file_path in tqdm(file_paths):
            if 'hdf5' in file_path:
                h5_file = h5py.File(file_path, 'r')
                tmp_ids.append(np.array(h5_file['ids']))
                # tmp_ids.append(torch.from_numpy(h5_file['ids'][:]))

                del h5_file

        print(f'Features collected: {len(tmp_ids)}')

        # Concatenate list of ids to one ids array
        ids = np.concatenate(tmp_ids, axis=0)
        # data = torch.cat(tmp_data)
        del tmp_ids

        del file_paths

        # Normalize to reduce numbers
        data = normalize(data)

        # Take the top k labels based on the rank
        top_k = 100  # First was 10

        # Load label embeddings
        label_embeddings = [
            os.path.join(root_directory, f)
            for f in os.listdir(root_directory)
            if f.endswith('.ptt') and MODEL in f
        ]
        print(f'Found {label_embeddings} embeddings')

        # Process loaded label files to gain label embeddings
        labels = []
        for label_embedding in label_embeddings:
            label_embeds = torch.load(label_embedding)
            for image in data:
                similarity = 100.0 * image @ label_embeds.T
                _, indices = similarity.topk(top_k)
                labels.append(indices)
            labels = torch.stack(labels)

        # Store data and ids in memory
        DATA = memory_data_storage(data, ids, labels)

        # Delete unused stuff
        del data
        del ids
        del labels

        # Store data and ids on hard drive
        os.makedirs(os.path.dirname(internal_storage), exist_ok=True)
        with open(internal_storage, 'wb+') as f:
            dill.dump(DATA, f)

        execution_time = time.time() - start_time
        print(f'Reading in and converting features took: {execution_time:.6f} secs')

    else:
        print('File already exists')


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Script to load stored features from hdf5 and convert it to one pickle file.'
    )

    # Define the arguments
    parser.add_argument('--base-dir', required=True, help='Base data directory')
    parser.add_argument('--model-name', required=True, help='Model name')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments and call the main function
    base_data_dir = args.base_dir
    model_name = args.model_name

    load_and_save_features(base_data_dir, model_name)
