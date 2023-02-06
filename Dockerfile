FROM ubuntu:20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget \
                       git \
    && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.12.0-Linux-x86_64.sh

WORKDIR /usr/src/app
COPY conda_requirements.txt ./
RUN conda install -c conda-forge --file conda_requirements.txt

COPY . .

ENTRYPOINT ["bash"]
