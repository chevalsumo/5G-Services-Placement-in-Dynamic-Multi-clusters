FROM ueransim-base AS builder
FROM ubuntu:20.04

 

LABEL maintainer="raoufkh <khichane.araouf@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ Europe/Paris
#add proxy to use apt
ENV http_proxy http://proxy.rd.francetelecom.fr:8080
ENV https_proxy http://proxy.rd.francetelecom.fr:8080
ENV no_proxy=172.18.0.1

 

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install iproute2 traceroute curl iputils-ping net-tools dnsutils file git nano build-essential

 

RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" -y < /dev/null \
&& (echo; echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"') >> /root/.bashrc \
&& eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)" \
&& brew install gcc k6 wrk hey

 

WORKDIR /ueransim

 

RUN mkdir config build

 

COPY --from=builder /ueransim/build/nr-ue ./build/
COPY --from=builder /ueransim/build/nr-cli ./build/
COPY --from=builder /ueransim/build/nr-binder ./build/
COPY --from=builder /ueransim/build/libdevbnd.so ./build/
COPY --from=builder /ueransim/config/* ./config/

 

VOLUME [ "/ueransim/config" ]

 

WORKDIR /ueransim/build

 

COPY wrapper.sh wrapper.sh
RUN chmod +x wrapper.sh

 

#ENTRYPOINT ["./wrapper.sh"]