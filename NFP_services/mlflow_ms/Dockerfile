FROM python:3.10.12-slim-bullseye

#
ENV http_proxy http://socks-proxy.rd.francetelecom.fr:8080
ENV https_proxy http://socks-proxy.rd.francetelecom.fr:8080
ENV no_proxy localhost,127.0.0.1
#



#RUN apt-get update
RUN apt-get update && apt-get install -y supervisor
#RUN apt-get install build-essential
#RUN apt-get install -y git
#RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
#ENV PATH="$HOME/.pyenv/bin:$PATH"

RUN pip install PyMySQL==0.9.3 && \   
    pip install mlflow==2.3.1 && \
    pip install watchdog && \
    pip install stable_baselines3 && \
    pip install gym && \
    pip install shimmy>=0.2.1
#RUN chmod 777 -R /mlflow
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

