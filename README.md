# RESTAPI for the Video Browser Showdown Challenge 

<!-- ![Your Project Logo](logo.png)  If applicable -->

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/your-repo-name.svg)](https://github.com/yourusername/your-repo-name/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/yourusername/your-repo-name.svg)](https://github.com/yourusername/your-repo-name/issues)

## Overview

Backend for the our submission Group at https://videobrowsershowdown.org/


## Table of Contents

- [RESTAPI for the Video Browser Showdown Challenge](#restapi-for-the-video-browser-showdown-challenge)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [API Documentation](#api-documentation)
    - [Text Query](#text-query)
    - [Image Query](#image-query)
    - [Get Video Image](#get-video-image)
  - [Running Tests](#running-tests)
  - [License](#license)

## Getting Started

See examples in the tests directory.
However, in general it is a RESTAPI with the following endpoints:  
/textQuery/  
/imageQuery/  
/getVideoImages/  

## Prerequisites

- **Python**: Python 3.7+ is required.

- **FastAPI & Uvicorn**

- **Pillow** (for image processing)

- **NumPy & pandas** (for data manipulation)

- **OpenAI GPT-3 & CLIP** (install as per their documentation)

- **python-multipart** (for handling multipart/form-data requests in FastAPI)

- **Transformers** (for natural language processing)

- **h5py** (for working with HDF5 data)

- **SciPy** (for scientific and technical computing)

- **Dill** (for object serialization)

- **Numba** (for JIT compiling Python functions)

Ensure Python is installed, create a virtual environment if needed, and install these libraries within your project environment for isolation.


## Installation

Step-by-step instructions on how to install and run your server.

```bash
# Clone the repository
git clone https://github.com/yourusername/your-repo-name.git

# Change directory
cd your-repo-name

# Use docker-compose
docker-compose up
```


## API Documentation

### Text Query

**Endpoint:** `/textQuery/`

**Method:** POST

**Description:** Accepts a JSON object with a text query and optional parameters. It retrieves a list of images based on the text query. Defaults: k=1000 (max=10000), get_embeddings=False

**Request Body Example:**
```json
{
  "query": "Your Text Query Here",
  "k": 5,
  "get_embeddings": false
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
    "label": null
  },
  {
    "uri": "another_image_uri",
    "rank": 2,
    "score": 0.92,
    "id": ["video_id", "frame_id"],
    "features": [0.2, 0.3, 0.4],
    "label": null
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
    "label": null
  },
  {
    "uri": "another_image_uri",
    "rank": 2,
    "score": 0.92,
    "id": ["video_id", "frame_id"],
    "features": [0.2, 0.3, 0.4],
    "label": null
  }
]
```

### Get Video Image

**Endpoint:** `/getVideoImage`

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


## Running Tests

You can run tests for your REST API server using shell scripts located in the `test` directory. Follow these steps to execute the tests:

```bash
# Change directory to the 'tests' folder
cd tests

# Run the tests using the shell script
./test_textQuery.sh
```
Ensure that you have set up any necessary test data or configurations before running the tests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.