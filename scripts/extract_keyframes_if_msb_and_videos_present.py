# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import re

import argparse
import subprocess

import time as time_

import pandas as pd

from pathlib import Path

from tqdm import tqdm


VIDEO_EXTENSIONS = ['.mp4', '.mov', '.m4v', '.avi', '.mpe']


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def millis():
    return int(round(time_.time() * 1000))


def get_video_path(file_name):
    for ext in VIDEO_EXTENSIONS:
        videopath = file_name + ext
        if os.path.exists(videopath):
            return videopath


def extract_keyframes(video_path, vf, kf_video_folder, video_name):
    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-vsync',
        '2',
        '-i',
        video_path,
        '-vf',
        'select=\'' + vf + '\'',
        f'{kf_video_folder}/{video_name}_%d.jpg',
    ]

    # extract list of keyframes
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print(f'ERROR in video {os.path.dirname(video_path)}: {err}')


def process_tsv(file, path_video, path_kf):
    # get name of the file with extension
    filename = os.path.basename(file)
    # get the name of the video without extension
    video_name = Path(filename).stem

    # reset variables
    vf = []

    # get the video folder and the output video keyframes paths
    video_path = get_video_path(os.path.join(path_video, filename.split('.tsv')[0]))
    if not video_path:
        video_path = get_video_path(
            os.path.join(
                path_video, filename.split('.tsv')[0], filename.split('.tsv')[0]
            )
        )
    print(video_path)
    if not video_path:
        return 0

    kf_video_folder = os.path.join(path_kf, video_name)

    # create folder for keyframes of the current video
    Path(kf_video_folder).mkdir(parents=True, exist_ok=True)

    # open file in pandas
    msb = pd.read_csv(file, delimiter='\t')

    # get the middleframe column
    mfs = msb['middleframe'].to_numpy(copy=True)
    ids = msb['id_visione'].to_numpy(copy=True)

    _, _, files = next(os.walk(kf_video_folder))
    found_files = sorted(files, key=natural_keys)

    expected_files = [f'{filename[:-4]}_{id}.jpg' for id in ids]

    if expected_files == found_files:
        return 1

    for mf in mfs:
        vf.append(f'eq(n\,{mf})')

    # join all the times of the frames to be extracted and extract keyframes
    vf = '+'.join(vf)
    extract_keyframes(video_path, vf, kf_video_folder, video_name)

    # scan keyframes folder for files
    _, _, files = next(os.walk(kf_video_folder))

    # rename keyframe files with the visione id
    files.sort(key=natural_keys)
    for i, kf_file in enumerate(files):
        file1 = re.sub('_\d+.jpg', f'_{ids[i]}.jpg', kf_file)
        p = Path(os.path.join(kf_video_folder, kf_file))
        p.rename(Path(p.parent, file1))

    return 1


def main():
    parser = argparse.ArgumentParser(
        description='Process video keyframes extraction from TSV files.'
    )
    parser.add_argument('-v', '--path_video', type=str, help='Path to the video files.')
    parser.add_argument('-m', '--path_msb', type=str, help='Path to the TSV files.')
    parser.add_argument(
        '-kf', '--path_kf', type=str, help='Path to save the keyframes.'
    )

    args = parser.parse_args()
    path_video = args.path_video
    path_msb = args.path_msb
    path_kf = args.path_kf

    # Initialize count_files and begin_time
    count_files = 0
    begin_time = millis()

    # Use pathlib.glob to handle paths
    file_paths = list(Path(path_msb).rglob('*.tsv'))
    file_paths.sort()
    for file in tqdm(file_paths):
        count_files += process_tsv(file, path_video, path_kf)

        if (count_files % 100) == 0:
            print(f'Analyzed {count_files} videos.')

    time_end = millis()
    print(f'Time to process {count_files} videos: {time_end - begin_time}')


if __name__ == "__main__":
    main()
