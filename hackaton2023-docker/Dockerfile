FROM python:3.10.12

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get -y install python3-opencv --no-install-recommends \
    python3-pip


COPY requirements.txt /tmp/requirements.txt
RUN pip3 --no-cache-dir install -r /tmp/requirements.txt
# RUN pip install onnxruntime onnx argparse

WORKDIR /mnt/
COPY . /app/

CMD ["/bin/bash", "-c", "python3 -u /app/main.py"]
