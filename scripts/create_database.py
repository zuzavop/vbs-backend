import argparse

import os
import dill
import h5py
import time

from tqdm import tqdm


def normalize(data):
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    return data


def load_and_save_features(DATABASE_ROOT, MODEL):
    # Specify the root directory you want to start walking from
    root_directory = os.path.join(DATABASE_ROOT, f'features-{MODEL}')

    # Internal storage
    internal_storage = os.path.join(
        DATABASE_ROOT, f'processed/pickled_files_{MODEL}.pkl'
    )
    if not os.path.exists(internal_storage):
        start_time = time.time()

        # Initialize an empty list to store file paths
        file_paths = []

        # Walk through the directory structure
        for folder, subfolders, files in os.walk(root_directory):
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
                tmp_data.append(np.array(h5_file['data']))

                del h5_file

        # Concatenate list of data to one data array
        data = np.concatenate(tmp_data, axis=0)
        del tmp_data

        tmp_ids = []
        for file_path in tqdm(file_paths):
            if 'hdf5' in file_path:
                h5_file = h5py.File(file_path, 'r')
                tmp_ids.append(np.array(h5_file['ids']))

                del h5_file

        print(len(tmp_ids))

        # Concatenate list of ids to one ids array
        ids = np.concatenate(tmp_ids, axis=0)
        del tmp_ids

        del file_paths

        # normalize to reduce numbers
        data = normalize(data)

        # store data and ids in memory
        set_data(data, ids)

        del data
        del ids

        # store data and ids on hard drive
        os.makedirs(os.path.dirname(internal_storage), exist_ok=True)
        with open(internal_storage, 'wb+') as f:
            dill.dump(DATA, f)

        execution_time = time.time() - start_time
        print(f'Reading in and converting features took: {execution_time:.6f} secs')

    else:
        print('File already exists')


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Script with two named arguments")

    # Define the arguments
    parser.add_argument('--base-dir', required=True, help="Base data directory")
    parser.add_argument('--model-name', required=True, help="Model name")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments and call the main function
    base_data_dir = args.base_dir
    model_name = args.model_name

    load_and_save_features(base_data_dir, model_name)
