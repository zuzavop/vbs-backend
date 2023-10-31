#!/bin/sh

server="http://localhost/api"
query='{"query": "A man with a hat", "k": 1000}'

curl -X POST -H "Content-Type: application/json" -d "$query" $server"/textQuery/"
