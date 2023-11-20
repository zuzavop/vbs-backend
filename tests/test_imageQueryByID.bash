#!/bin/bash

server="http://localhost/api"
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi
echo "$server"
query='{"item_id": "00001_6", "k": 1000, "add_features": "0"}'
query='{"video_id": "00001", "frame_id": "6", "k": 1000, "add_features": "0"}'

curl -X POST -H "Content-Type: application/json" -d "$query" $server"/imageQueryByID/"
