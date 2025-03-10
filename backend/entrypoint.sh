#!/bin/bash
set -e

echo "Așteptăm PostgreSQL să pornească..."
sleep 5

echo "Populăm baza de date..."
python db.py || { echo "Eroare la popularea bazei de date"; exit 1; }

echo "Pornim aplicația..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
