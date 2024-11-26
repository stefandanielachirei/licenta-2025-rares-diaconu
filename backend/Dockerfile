# Folosim o imagine de bază Python 3.12 Alpine
FROM python:3.12-alpine

# Setăm variabile de mediu
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Instalăm dependențele necesare pentru PostgreSQL și compilarea pachetelor Python
RUN apk update && apk add --no-cache \
    gcc \
    musl-dev \
    postgresql-dev \
    libpq \
    linux-headers

# Setăm directorul de lucru în container
WORKDIR /app

# Copiem fișierul requirements.txt și instalăm pachetele
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiem codul aplicației în container
COPY . /app/

# Expunem portul 5000
EXPOSE 8000

# Comanda care rulează aplicația FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
