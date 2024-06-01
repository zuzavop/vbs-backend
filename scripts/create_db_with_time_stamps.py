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
    METADATA = None

    def __init__(self, new_data=None, new_ids=None, new_labels=None, new_time=None, new_metadata=None):
        self.DATA = new_data
        self.IDS = new_ids
        self.LABELS = new_labels
        self.TIME = new_time
        self.METADATA = new_metadata


# Function to extend a database with timestamps obtained from corresponding msb files
def extend_db_with_time_stamps(db_dir, msb_dir):
    # Get all database files (ending with .pkl) excluding those with 'db_time' in the name
    pkl_files = [
        file
        for file in os.listdir(db_dir)
        if file.endswith('.pkl') and 'db_time' not in file and 'db_metadata' not in file
    ]

    # Get all msb files (ending with .tsv)
    msb_files = [file for file in os.listdir(msb_dir) if file.endswith('.tsv')]

    # Dictionary to store loaded msb files for efficient reuse
    loaded_msb_files = {}

    # Loop through all the database files and check if timestamps are present
    for file in pkl_files:
        print(f'Adding to database file: {file}')
        internal_storage = os.path.join(db_dir, file)

        try:
            print(f'Opening file {internal_storage}')
            with open(internal_storage, 'rb') as f:
                data = dill.load(f)

            # Create a separate file for timestamps based on the original database file
            time_stamp_storage = f'{internal_storage[:-4]}_time.pkl'

            # Check conditions for timestamp existence or validity in the data
            if (
                not os.path.exists(time_stamp_storage)
                or data.TIME is None
                or data.TIME == ''
                or not hasattr(data, 'TIME')
            ):
                time_stamps = []

                print(f'Starting to add time stamps to the database')

                # Extract video IDs and frame IDs from the database
                ids = np.array(data.IDS).astype(str)

                # Iterate over unique IDs, extracting corresponding timestamps from msb files
                for id in tqdm(ids):
                    splits = id.split('-', 1)
                    if len(splits) < 2:
                        splits = id.split('_', 1)
                    video_id, frame_id = splits
                    msb_file = video_id + '.tsv'

                    # Check if the msb file is available
                    if msb_file in msb_files:
                        msb_file = os.path.join(msb_dir, msb_file)

                        try:
                            # Load the msb file into a DataFrame
                            if msb_file in loaded_msb_files:
                                df = loaded_msb_files[msb_file]
                            else:
                                df = pd.read_csv(msb_file, delimiter='\t')
                                loaded_msb_files[msb_file] = df

                            # Select the timestamps corresponding to the given frame ID
                            selected_time_stamps = df[
                                df['id_visione'] == float(frame_id)
                            ]

                            if selected_time_stamps.empty:
                                selected_time_stamps = df[
                                    df['id_visione'] == int(frame_id)
                                ]

                            if selected_time_stamps.empty:
                                selected_time_stamps = df[df['id_visione'] == frame_id]

                            # Append the extracted timestamps to the list
                            time_stamps.append(
                                [
                                    id,
                                    selected_time_stamps['middletime'].item() * 1000,
                                    selected_time_stamps['starttime'].item() * 1000,
                                    selected_time_stamps['endtime'].item() * 1000,
                                ]
                            )

                        except Exception as e:
                            print(f'Could not load {id} with {e}')

                print(f'Found: {len(time_stamps)}, Ids: {len(ids)}')

                # Update the original data with the new timestamp storage path
                data.TIME = time_stamp_storage

                # Write the updated data back to the original database file
                with open(internal_storage, 'wb') as f:
                    dill.dump(data, f)
                print(f'Successful writing file {internal_storage}')

                # Write the extracted timestamps to the newly created timestamp storage file
                with open(time_stamp_storage, 'wb') as f:
                    dill.dump(time_stamps, f)
                print(f'Successful writing file {time_stamp_storage}')

                # Cleanup: delete temporary variables
                del time_stamps
                del data

            else:
                # If timestamps are already present, inform the user
                print(f'Already contains time stamps {internal_storage}')

        except Exception as e:
            # Handle exceptions during file processing
            print(f'Failed with file {internal_storage} and {e}')

    # Cleanup: delete loaded msb files
    del loaded_msb_files


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
