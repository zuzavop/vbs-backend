#!/bin/bash

server="http://localhost/api"
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi
echo "$server"
query='{"query": "A man with a hat", "k": 100, "add_features": 0, "max_labels": 10}'

curl -X POST -H "Content-Type: application/json" -d "$query" $server"/textQuery/"
