#!/bin/sh

query='{"item_id": "00001_6", "k": 1000}'

curl -X POST -H "Content-Type: application/json" -d "$query" http://localhost:8000/getVideoFrames/
