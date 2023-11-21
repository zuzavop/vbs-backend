#!/bin/bash

server="http://localhost/api"
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi
echo "$server"
query='{"k": 1000, "add_features": 1}'

curl -F "image=@image.jpg" -F "query_params=$query" $server"/imageQuery/"
