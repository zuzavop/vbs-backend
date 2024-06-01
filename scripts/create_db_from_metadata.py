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

    def __init__(self, new_data=None, new_ids=None, new_labels=None, new_time=None, new_metadata=None):
        self.DATA = new_data
        self.IDS = new_ids
        self.LABELS = new_labels
        self.TIME = new_time
        self.METADATA = new_metadata


def extend_db_with_metadata(db_dir, metadata_path):
    # Get all database files (ending with .pkl) excluding those with 'db_time' in the name
    pkl_files = [
        file
        for file in os.listdir(db_dir)
        if file.endswith('.pkl') and 'db_time' not in file and 'db_metadata' not in file
    ]
    
    try:
        metadata_file = pd.read_csv(metadata_path, delimiter=',')
        metadata_file.set_index('minute_id', inplace=True)
    except Exception as e:
        print(f'Could not load metadata file with {e}')
        return
    
    for file in pkl_files:
        print(f'Adding to database file: {file}')
        internal_storage = os.path.join(db_dir, file)
        
        try:
            print(f'Opening file {internal_storage}')
            with open(internal_storage, 'rb') as f:
                data = dill.load(f)

            # Create a separate file for timestamps based on the original database file
            metadata_storage = f'{internal_storage[:-4]}_metadata.pkl'

            # Check conditions for timestamp existence or validity in the data
            if (
                not os.path.exists(metadata_storage)
                or data.METADATA is None
                or data.METADATA == ''
                or not hasattr(data, 'METADATA')
            ):
                metadata = []

                print(f'Starting to add metadata to the database')

                # Extract video IDs and frame IDs from the database
                ids = np.array(data.IDS).astype(str)

                # Iterate over unique IDs, extracting corresponding metadata from metadata file
                for id in tqdm(ids):
                    minute_id = id[:13]

                    try:
                        # Select the metadata corresponding to the given frame ID
                        selected_metadata = metadata_file.loc[minute_id]
                        
                        date = datetime(int(id[:4]), int(id[4:6]), int(id[6:8]))

                        # Append the extracted metadata to the list
                        metadata.append(
                            [
                                id,
                                selected_metadata['song name'].item(),
                                selected_metadata['album name'].item(),
                                selected_metadata['artist name'].item(),
                                selected_metadata['utc_time'].item(),
                                selected_metadata['local_time'].item(),
                                selected_metadata['latitude'].item(),
                                selected_metadata['longitude'].item(),
                                selected_metadata['semantic_name'].item(),
                                selected_metadata['time_zone'].item(),
                                id.split('_')[1][:2],
                                date.weekday(),
                            ]
                        )
                    except Exception as e:
                        print(f'Could not find metadata for {minute_id} and {e}')

                print(f'Found: {len(metadata)}, Ids: {len(ids)}')

                # Update the original data with the new timestamp storage path
                data.METADATA = metadata_storage

                # Write the updated data back to the original database file
                with open(internal_storage, 'wb') as f:
                    dill.dump(data, f)
                print(f'Successful writing file {internal_storage}')

                # Write the extracted timestamps to the newly created timestamp storage file
                with open(metadata_storage, 'wb') as f:
                    dill.dump(metadata, f)
                print(f'Successful writing file {metadata_storage}')

                # Cleanup: delete temporary variables
                del metadata
                del data

            else:
                # If timestamps are already present, inform the user
                print(f'Already contains metadata {internal_storage}')

        except Exception as e:
            # Handle exceptions during file processing
            print(f'Failed with file {internal_storage} and {e}')

    # Cleanup: delete loaded files
    del metadata_file

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to extend all pickled databases with metadata')
    parser.add_argument('--db_dir', type=str, help='Directory containing the database files')
    parser.add_argument('--metadata_file', type=str, help='Path to the metadata file')
    args = parser.parse_args()
    
    db_dir = os.path.join(args.db_dir, 'processed')
    metadata = args.metadata_file

    extend_db_with_metadata(db_dir, metadata)