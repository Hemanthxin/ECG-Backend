#!/bin/bash

echo "Starting ECG Backend..."

uvicorn main:app --host 0.0.0.0 --port $PORT
