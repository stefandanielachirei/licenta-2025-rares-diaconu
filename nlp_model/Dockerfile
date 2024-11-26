FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch==1.13.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

#COPY ./backend /app
COPY ./model /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
