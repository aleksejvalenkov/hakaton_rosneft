DOCKER_NAME := hackaton2023
WORK_DIR := /home/alex/Documents/hackatons/DATA_SETS

build:
	docker build -f Dockerfile -t ${DOCKER_NAME} .

build_cuda:
	docker build -f Dockerfile_Cuda -t ${DOCKER_NAME} .

build_pytorch:
	docker build -f Dockerfile_Pytorch -t ${DOCKER_NAME} .

run:
	docker run -it --rm -v ${WORK_DIR}:/mnt/ ${DOCKER_NAME}

run_cuda: 
	docker run -it --rm --runtime=nvidia -v ${WORK_DIR}:/mnt/ ${DOCKER_NAME}

shell: 
	docker run -it --rm -v ${WORK_DIR}:/mnt/ ${DOCKER_NAME} bash
