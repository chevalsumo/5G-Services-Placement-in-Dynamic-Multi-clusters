FROM dtzar/helm-kubectl
#FROM python:3.10.12-slim-bullseye

ENV http_proxy http://socks-proxy.rd.francetelecom.fr:8080
ENV https_proxy http://socks-proxy.rd.francetelecom.fr:8080

RUN apk update

RUN apk add --update --no-cache curl py-pip
#RUN apk add py3-pip
#RUN apk add --update py-pip
RUN apk add --no-cache py3-numpy py3-scikit-learn
COPY ./submariner.sh /home/submariner.sh 
COPY ./requirements.txt /home/requirements.txt

RUN bash /home/submariner.sh
#RUN export PATH=$PATH:~/.local/bin
ENV PATH "$PATH:~/.local/bin"
RUN pip install -r /home/requirements.txt

ENV https_proxy ""
ENV http_proxy ""