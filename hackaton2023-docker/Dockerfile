FROM ubuntu:bionic
RUN apt-get update && apt-get -y install python3-opencv --no-install-recommends
WORKDIR /mnt/
COPY . /app/
# COPY ${WORK_DIR}/input.webm /mnt/
CMD ["/usr/bin/python3", "/app/main.py"]
