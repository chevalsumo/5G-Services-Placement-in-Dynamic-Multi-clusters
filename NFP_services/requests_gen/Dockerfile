FROM python:3.10.12-slim-bullseye


ENV http_proxy http://socks-proxy.rd.francetelecom.fr:8080
ENV https_proxy http://socks-proxy.rd.francetelecom.fr:8080
ENV no_proxy localhost,127.0.0.1


COPY ./requirements.txt /home/requirements.txt
COPY ./api.py /home/api.py

WORKDIR /home

RUN pip install -r /home/requirements.txt

