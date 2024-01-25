#!/bin/bash

server="http://localhost/api"
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi
echo "$server"
query='{"item_id": "00001_6", "dataset": "V3C", "k": -1, "add_features": 0}'
query='{"item_id": "LHE59_001", "dataset": "VBSLHE", "k": -1, "add_features": 0}'

curl -X POST -H "Content-Type: application/json" -d "$query" $server"/getVideoFrames/"
