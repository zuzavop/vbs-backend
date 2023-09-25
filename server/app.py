from fastapi import FastAPI, File, Query, Body, UploadFile

import logging

from PIL import Image
from io import BytesIO

import features as fs
import database as db
import logger as l


# Create an instance of the FastAPI class
app = FastAPI()

db.load_features()


# Define the 'textquery' route
@app.post('/textQuery/')
async def text_query(query_params: dict, get_embeddings: bool = False):
    '''
    Get a list of images based on a text query.
    '''
    l.logger.info(query_params)
    query = query_params.get("query", "")
    k = query_params.get("k", 10)

    # Call the function to retrieve images
    images = fs.get_images_by_text_query(query, k)
    if not get_embeddings:
        images = [[x[0], x[2]] for x in images]

    return {'images': images}


# Define the 'imagequery' route
@app.post('/imageQuery/')
async def image_query(
    image: UploadFile,
    k: int = Query(10, title='Number of Images'),
    get_embeddings: bool = False,
):
    '''
    Get a list of images based on an image query.
    '''
    # Call the function to retrieve images
    try:
        # Read the uploaded image file
        image_data = await image.read()

        # Open the image using Pillow (PIL)
        uploaded_image = Image.open(BytesIO(image_data))

        # Perform image processing here (e.g., resizing, filtering, etc.)
        # For this example, we'll just return the image dimensions.
        image_width, image_height = uploaded_image.size

        images = fs.get_images_by_image_query(uploaded_image, k)
        if not get_embeddings:
            images = [[x[0], x[2]] for x in images]

        return {'images': images}
    except Exception as e:
        return {"error": str(e)}


# Define the 'getvideo' route
@app.get('/getVideo')
async def get_video_images(
    video_id: str = Query(..., title='Video ID'),
    k: int = Query(10, title='Number of Images'),
):
    '''
    Get a list of video images based on a video ID.
    '''
    # Call the function to retrieve video images
    video_images = fs.get_video_images_by_id(video_id, k)

    return {'video_id': video_id, 'k': k, 'video_images': video_images}


# Define a route and its handler function
@app.get('/')
async def read_root():
    return {'message': 'Server is running!'}


# @app.on_event("startup")
# async def startup_event():
#     logger = logging.getLogger('uvicorn.access')
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
#     logger.addHandler(handler)


# Run the FastAPI application
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
