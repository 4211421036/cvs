#!/bin/bash

echo "Starting Flask Server..."
docker build -t python-backend .
docker run -d -p 5000:5000 python-backend
