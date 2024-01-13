#!/bin/bash

server="http://localhost/api"
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi
echo "$server"
query='{"dataset": "VBSLHE", "add_features": 0}'

curl -X GET -H "Content-Type: application/json" -d "$query" $server"/getRandomFrame/"
