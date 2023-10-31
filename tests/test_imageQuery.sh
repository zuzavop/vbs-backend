#!/bin/sh

query='{"k":1000}'

curl -F "image=@image.jpg" -F "query_params=$query" http://localhost/api/imageQuery/
