#!/bin/bash

server="http://localhost/api"
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi
echo "$server"
query='{"query": "A man with a hat", "k": 1000, "add_features": 1, "max_labels": 10}'

curl -X POST -H "Content-Type: application/json" -d "$query" $server"/textQuery/"
