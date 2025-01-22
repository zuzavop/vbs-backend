# RESTAPI for the Video Browser Showdown Challenge 

<!-- ![Your Project Logo](logo.png)  If applicable -->

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/merowech/vbs-backend.svg)](https://github.com/merowech/vbs-backend/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/merowech/vbs-backend.svg)](https://github.com/merowech/vbs-backend/issues)

## Overview

Backend for our submission at the https://videobrowsershowdown.org/ VBS challenge.
The VBS is an international video content search competition that evaluates the state-of-the-art of interactive video retrieval systems. It is performed annually as a special event at the International Conference on MultiMedia Modeling (MMM) since 2012. It aims at pushing research on large-scale video retrieval systems that are effective, fast, and easy to use for content search scenarios that are truly relevant in practice (e.g., known-item search in an ever-increasing video archive, as nowadays ubiquitous in many domains of our digital world).

The VBS usually consists of an expert session and a novice session. In the expert session the developers of the systems themselves try to solve different types of content search queries that are issued in an ad-hoc manner. Although the dataset itself is available to the researchers several months before the actual competition, the queries are unknown in advance and issued on-site. In the novice session volunteers from the MMM conference audience (without help from the experts) are required to solve another set of tasks. This should ensure that the interactive video retrieval tools do not only improve in terms of retrieval performance but also in terms of usage (i.e., ease-of-use).

There are different types of queries:

 - **Known-Item Search (KIS):** a single video clip (20 secs long) is randomly selected from the dataset and visually presented with the projector on-site. The participants need to find exactly the single instance presented. Another task variation of this kind is textual KIS, where instead of a visual presentation, the searched segment is described only by text given by the moderator (and presented as text via the projector).
 - **Ad-hoc Video Search (AVS):** here, a rather general description of many shots is presented by the moderator (e.g., „Find all shots showing cars in front of trees“) and the  participants need to find as many correct examples (instances) according to the description.
 - **Question-Answering (QA):** the participants need to answer a question about the content of the video dataset. The questions are usually of the form „Which color has the object in the video, which was shown?“ or „What was written on the sign in the video?“.
Each query has a time limit (e.g., 5-7 minutes) and is rewarded on success with a score that depends on several factors: the required search time, the number of false submissions (which are penalized), and the number of different instances found for AVS tasks. For the latter case it is also considered, how many different ‚ranges‚ were submitted for an AVS tasks. For example, many different but temporally close shots in the same video count much less than several different shots from different videos.

## Table of Contents

- [RESTAPI for the Video Browser Showdown Challenge](#restapi-for-the-video-browser-showdown-challenge)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Setting up the Environment](#setting-up-the-environment)
  - [API Documentation](#api-documentation)
    - [Text Query](#text-query)
    - [Image Query](#image-query)
    - [Get Video Image](#get-video-image)
    - [Temporal Query](#temporal-query)
    - [Filter](#filter)
  - [Running Tests](#running-tests)
  - [Possible Datasets](#possible-datasets)
  - [License](#license)
  - [Reference](#reference)

## Getting Started

See examples in the tests directory.
However, in general it is a RESTAPI with the following endpoints:  
/textQuery/  
/imageQuery/  
/imageQueryByID/  
/getVideoImages/
/temporalQuery/
/filter/

## Prerequisites

- **Python**: Python 3.7+ is required.

- **Nginx** (for general route handling)

- **FastAPI & Uvicorn** (for request handling)

- **Pillow** (for image processing)

- **NumPy & pandas** (for data manipulation)

- **OpenAI GPT-3 & CLIP** (install as per their documentation)

- **python-multipart** (for handling multipart/form-data requests in FastAPI)

- **Transformers** (for natural language processing)

- **h5py** (for working with HDF5 data)

- **SciPy** (for scientific and technical computing)

- **Dill** (for object serialization)

Ensure Python is installed, create a virtual environment if needed, and install these libraries within your project environment for isolation.


## Installation

Step-by-step instructions on how to install and run your server.

```bash
# Clone the repository
git clone https://github.com/merowech/vbs-backend.git

# Change directory
cd vbs-backend

# Use docker-compose
docker-compose up --build
```

If new model like `ViT-SO400M-14-SigLIP-384` will be used, you need to download it first and add it to the `model` folder.

### Setting up the Environment

Set $DATA_PATH to the path where the data is stored. Create a folder for each dataset and add the data to it. Each type of data needs to be stored in a separate folder and named properly. The data should be stored in the following structure:

```
$DATA_PATH
│
└───name_of_dataset
│   └───features-version_of_clip.tar.gz
│   └───msb.tar.gz
│   └───metadata.csv
│   └───nounlist.txt

```

The `features-version_of_clip.tar.gz` file contains the features extracted from the dataset using the CLIP model. The `msb.tar.gz` file contains the timestamps of the dataset. The `metadata.csv` file contains the metadata of the dataset. The `nounlist.txt` file contains the list of nouns in the dataset that will be automatically extracted. If another metadata file type are used appropriate python script should be created to add the metadata to the database and `scripts/preprocess.bash` should be updated to reflect the changes.


## API Documentation

In general, there are several default parameters available for every query:  
Defaults: `{"k": 1000, "dataset": "V3C", "model": "clip-laion", "max_labels": 10, "add_features": 0, "speed_up": 1}`  
`"add_features"` adds features to the returning json depending on the `"dataset"` and `"model"`.  
`"speed_up"` enables a download of the json file which speeds up the whole process.  

### Text Query

**Endpoint:** `/textQuery/`

**Method:** POST

**Description:** Accepts a JSON object with a text query and optional parameters. It retrieves a list of images based on the text query.   

**Request Body Example:**
```json
{
  "query": "Your Text Query Here",
  "k": 5,
  "dataset": "Dataset Name",
  "model": "Model Name",
  "add_features": false,
  "speed_up": true,
}
```
  
See `tests\test_textQuery.sh` for an example.

**Response Example:**
```json
[
  {
    "uri": "image_uri",
    "rank": 1,
    "score": 0.95,
    "id": ["video_id", "frame_id"],
    "features": [0.1, 0.2, 0.3],
    "label": [5, 10, 2, 3, 1],
    "time": ["id", 1450.0, 1350.0, 1550.0],
  },
  {
    "uri": "another_image_uri",
    "rank": 2,
    "score": 0.92,
    "id": ["video_id", "frame_id"],
    "features": [0.2, 0.3, 0.4],
    "label": [7, 4, 3, 9, 10],
    "time": ["id", 1450.0, 1350.0, 1550.0],
  }
]
```

### Image Query

**Endpoint:** `/imageQuery/`

**Method:** POST

**Description:** Accepts an image file upload and optional parameters. It retrieves a list of images based on the uploaded image query.

**Request Example:**

See `tests\test_imageQuery.sh` for an example.

**Response Example:**
```json
[
  {
    "uri": "image_uri",
    "rank": 1,
    "score": 0.95,
    "id": ["video_id", "frame_id"],
    "features": [0.1, 0.2, 0.3],
    "label": [5, 10, 2, 3, 1],
    "time": ["id", 1450.0, 1350.0, 1550.0],
  },
  {
    "uri": "another_image_uri",
    "rank": 2,
    "score": 0.92,
    "id": ["video_id", "frame_id"],
    "features": [0.2, 0.3, 0.4],
    "label": [7, 4, 3, 9, 10],
    "time": ["id", 1450.0, 1350.0, 1550.0],
  }
]
```

### Get Video Image

**Endpoint:** `/getVideoFrames`

**Method:** GET

**Description:** Accepts a video ID and frame ID as parameters. It retrieves a specific video image based on the provided IDs.

**Request Example:**
See `tests\test_getVideoImage.sh` for an example.

**Response Example:**
```json
{
  "video_id": "YourVideoIDHere",
  "frame_id": "YourFrameIDHere",
  "video_image": "image_data_here"
}
```

### Temporal Query

**Endpoint:** `/temporalQuery/`

**Method:** POST

**Description:** Accepts a JSON object with a temporal query and optional parameters. It retrieves a list of images based on the temporal query.

**Request Body Example:**
```json
{
  "query": "Your First Part of Text Query Here",
  "query2": "Your Second Part od Text Query Here",
  "k": 5,
  "dataset": "Dataset Name",
  "model": "Model Name",
  "add_features": false,
  "speed_up": true,
}
```

**Response Example:**
```json
[
  {
    "uri": "image_uri",
    "rank": 1,
    "score": 0.95,
    "id": ["video_id", "frame_id"],
    "features": [0.1, 0.2, 0.3],
    "label": [5, 10, 2, 3, 1],
    "time": ["id", 1450.0, 1350.0, 1550.0],
  },
  {
    "uri": "another_image_uri",
    "rank": 2,
    "score": 0.92,
    "id": ["video_id", "frame_id"],
    "features": [0.2, 0.3, 0.4],
    "label": [7, 4, 3, 9, 10],
    "time": ["id", 1450.0, 1350.0, 1550.0],
  }
]
```

### Filter

**Endpoint:** `/filter/`

**Method:** POST

**Description:** Accepts a JSON object with a list of image URIs and optional parameters. It retrieves a list of images based on the provided URIs.

**Request Body Example:**
```json
{
  "filter": { "weekday": "Monday", "hour": "12-15" },
  "k": 5,
  "dataset": "Dataset Name",
  "model": "Model Name",
  "add_features": false,
  "speed_up": true,
}
```

**Response Example:**
```json
[
  {
    "uri": "image_uri",
    "rank": 1,
    "score": 0,
    "id": ["video_id", "frame_id"],
    "features": [0.1, 0.2, 0.3],
    "label": [5, 10, 2, 3, 1],
    "time": ["id", 1450.0, 1350.0, 1550.0],
  },
  {
    "uri": "another_image_uri",
    "rank": 2,
    "score": 0,
    "id": ["video_id", "frame_id"],
    "features": [0.2, 0.3, 0.4],
    "label": [7, 4, 3, 9, 10],
    "time": ["id", 1450.0, 1350.0, 1550.0],
  }
]
```


## Running Tests

You can run tests for your REST API server using shell scripts located in the `test` directory. Follow these steps to execute the tests:

```bash
# Change directory to the 'tests' folder
cd tests

# Run the tests using the shell script
./test_textQuery.sh
```

Ensure that you have set up any necessary test data or configurations before running the tests.

## Possible Datasets

Currently supported datasets for the VBS challenge:

 - V3C (https://zenodo.org/records/8188570)
 - MVK (https://zenodo.org/records/8355037)
 - VBSLHE (https://zenodo.org/records/10013329)
 - LSC (http://lifelogsearch.org/lsc/lsc_data/)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Reference

```
@Article{lokoc_prak_2024,
  author  = {Jakub Loko\v{c} and Zuzana Vopálková and Michael Stroh and Raphael Buchmueller and Udo Schlegel},
  journal = {VBS 2024 at MMM 2024},
  title   = {{PraK Tool: An Interactive Search Tool Based on Video Data Services}},
  year    = {2024},
}
```

```
@inproceedings{vopalkova2024searching,
  title={Searching temporally distant activities in lifelog data with prak tool V2},
  author={Vop{\'a}lkov{\'a}, Zuzana and Yaghob, Jakub and Stroh, Michael and Schlegel, Udo and Lokoc, Jakub},
  booktitle={Proceedings of the 7th Annual ACM Workshop on the Lifelog Search Challenge},
  pages={111--116},
  year={2024}
}
```

```
@inproceedings{stroh2025prak,
  title={PraK Tool V3: Enhancing Video Item Search Using Localized Text and Texture Queries},
  author={Stroh, Michael and Kloda, Vojt{\v{e}}ch and Verner, Benjamin and Vop{\'a}lkov{\'a}, Zuzana and Buchm{\"u}ller, Raphael and J{\"a}ckl, Bastian and Hajko, Jakub and Loko{\v{c}}, Jakub},
  booktitle={International Conference on Multimedia Modeling},
  pages={326--333},
  year={2025},
  organization={Springer}
}
```
