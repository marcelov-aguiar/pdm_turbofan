FROM python:3.8.9-slim
RUN apt update && \
    apt install -y git
RUN pip install --upgrade pip
ADD ./docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt