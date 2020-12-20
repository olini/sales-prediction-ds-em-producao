FROM python:3.8-slim
COPY . /app
WORKDIR /app
RUN apt-get update && \
    apt-get install libgomp1
RUN pip3 install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:$PORT wsgi