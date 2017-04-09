FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

WORKDIR /dockerwork

RUN set -x && apt-get update && apt-get install -y\
    python3\
    python3-pip\
    vim\
    git\
    && pip3 install\
    chainer==1.22.0\
    git+https://github.com/risuoku/wataru.git\

    # clean 
    && apt-get clean && rm -rf /var/lib/apt/lists/*
