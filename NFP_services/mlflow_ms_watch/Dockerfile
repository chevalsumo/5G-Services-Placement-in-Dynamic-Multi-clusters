FROM python:3.10.12-slim-bullseye

#
ENV http_proxy http://socks-proxy.rd.francetelecom.fr:8080
ENV https_proxy http://socks-proxy.rd.francetelecom.fr:8080
ENV no_proxy localhost,127.0.0.1
#



#RUN apt-get update
RUN apt-get update && apt-get install -y supervisor



RUN pip install watchdog && \
    pip install numpy && \
    pip install pyyaml && \
    pip install requests

ENV http_proxy=
ENV https_proxy=