#!/bin/sh

server="http://localhost:8000"
query='{"query": "A man with a hat", "k": 5}'

curl -X POST -H "Content-Type: application/json" -d "$query" $server"/textQuery/"
