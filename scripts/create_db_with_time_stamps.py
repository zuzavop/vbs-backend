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


# Class structure to store data, indices, labels, and the time
class memory_data_storage:
    DATA = None
    IDS = None
    LABELS = None
    TIME = None

    def __init__(self, new_data=None, new_ids=None, new_labels=None, new_time=None):
        self.DATA = new_data
        self.IDS = new_ids
        self.LABELS = new_labels
        self.TIME = new_time


def extend_db_with_time_stamps(db_dir, msb_dir):
    # Get all database files and all msb files
    pkl_files = [
        file
        for file in os.listdir(db_dir)
        if file.endswith('.pkl') and 'db_time' not in file
    ]
    msb_files = [file for file in os.listdir(msb_dir) if file.endswith('.tsv')]

    loaded_msb_files = {}

    # Loop through all the database files and check if the time stamps are there
    for file in pkl_files:
        print(f'Adding to database file: {file}')
        internal_storage = os.path.join(db_dir, file)

        try:
            print(f'Opening file {internal_storage}')
            with open(internal_storage, 'rb') as f:
                data = dill.load(f)
            time_stamp_storage = f'{internal_storage[:-4]}_time.pkl'
            if (
                not os.path.exists(time_stamp_storage)
                or data.TIME is None
                or data.TIME == ''
                or not hasattr(data, 'TIME')
            ):
                time_stamps = []

                print(f'Starting to add time stamps to the database')
                ids = np.array(data.IDS).astype(str)
                for id in tqdm(ids):
                    splits = id.split('-', 1)
                    if len(splits) < 2:
                        splits = id.split('_', 1)
                    video_id, frame_id = splits
                    msb_file = video_id + '.tsv'
                    if msb_file in msb_files:
                        msb_file = os.path.join(msb_dir, msb_file)
                        try:
                            if msb_file in loaded_msb_files:
                                df = loaded_msb_files[msb_file]
                            else:
                                df = pd.read_csv(msb_file, delimiter='\t')
                                loaded_msb_files[msb_file] = df
                            selected_time_stamps = df[
                                df['id_visione'] == float(frame_id)
                            ]
                            time_stamps.append(
                                [
                                    id,
                                    selected_time_stamps['middletime'].item() * 1000,
                                    selected_time_stamps['starttime'].item() * 1000,
                                    selected_time_stamps['endtime'].item() * 1000,
                                ]
                            )
                        except:
                            print(f'Could not load {id}')

                print(f'Found: {len(time_stamps)}, Ids: {len(data.IDS)}')
                data.TIME = time_stamp_storage
                with open(internal_storage, 'wb') as f:
                    dill.dump(data, f)
                with open(time_stamp_storage, 'wb') as f:
                    dill.dump(time_stamps, f)
                print(f'Successful writing file {time_stamp_storage}')
            else:
                print(f'Already contains time stamps {internal_storage}')
        except Exception as e:
            print(f'Failed with file {internal_storage} and {e}')


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Script to extend all pickled databases with the time stamps of the extracted keyframes.'
    )

    # Define the arguments
    parser.add_argument('--db-dir', required=True, help='Database directory')
    parser.add_argument('--msb-dir', required=True, help='MSB files directory')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments and call the main function
    db_dir = os.path.join(args.db_dir, 'processed')
    msb_dir = args.msb_dir

    print(f'Got the following directories: {db_dir} and {msb_dir}')

    extend_db_with_time_stamps(db_dir, msb_dir)
