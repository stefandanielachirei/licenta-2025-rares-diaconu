#!/bin/bash
set -e

echo "Waiting for PostgreSQL to launch..."
sleep 5

echo "Populating the database..."
python db.py || { echo "Error at populating the database"; exit 1; }

echo "Starting the app..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
