FROM python:3.10-slim-bullseye

RUN apt-get update -y; \
    apt-get install libgomp1 -y; \
    apt-get install iputils-ping -y; \
    apt-get install libgl1-mesa-glx -y; \
    apt-get install procps -y; \
    apt-get install pkg-config -y; \
    apt-get install libhdf5-dev -y; \
    # gcc for h5py
    apt-get install gcc -y; \
    rm -rf /var/lib/apt/lists/*; \
    pip install --no-cache-dir --upgrade pip

