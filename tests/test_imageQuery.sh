#!/bin/sh

curl -F "image=@image.jpg" -F "k=1000" http://localhost:8000/imageQuery/
