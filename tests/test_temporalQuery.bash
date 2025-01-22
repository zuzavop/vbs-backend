#!/bin/bash

server="http://localhost/api"
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi
echo "$server"
query='{"query": "A diver in blue suit", "query2": "A turtle", "k": 100, "dataset": "MVK", "add_features": 0, "max_labels": 10}'

curl -X POST -H "Content-Type: application/json" -d "$query" $server"/temporalQuery/"
