#!/usr/bin/python3

import cv2
import numpy as np
from utils import mask2rle
import time
from hybridnets import HybridNets, optimized_model

# Файлы ввода-вывода
INPUT_FILE = 'input.webm'
OUTPUT_FILE = "output.csv"

DEBUG = False

segmentation_colors = np.array([[0,    0,    0],
								[125,  125,  125],
								[255,  255,   255]], dtype=np.uint8)

def get_mask_img(seg_map, start_img):
	img = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype = np.uint8)
	img[seg_map==1] = segmentation_colors[seg_map[seg_map==1]]
	img = cv2.resize(img, (start_img.shape[1], start_img.shape[0]))
	return img

def get_mask_arr(seg_map, start_img):
	masked_img = get_mask_img(seg_map, start_img)
	lower = np.array([120,120,120])
	upper = np.array([130,130,130])
	mask = cv2.inRange(masked_img, lower, upper)
	return mask

def get_mask_arr_b(mask):
	mask_b = np.where(mask == 255, 1, mask)
	return mask_b

# Считать видео
cap = cv2.VideoCapture(INPUT_FILE)
frame_number = 0

if cap.isOpened() == False:
    print("Error opening video file")

# Подготовить запись файла
out = open(OUTPUT_FILE, "w")
out.write("ImageID,EncodedPixels\n")


# Initialize road detector
model_path = "/app/models/hybridnets_512x640/hybridnets_512x640.onnx"
anchor_path = "/app/models/hybridnets_512x640/anchors_512x640.npy"
optimized_model(model_path) # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.7, iou_thres=0.5)
if DEBUG:
    cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)	

# Основной цикл
while True:
    ret, frame = cap.read()
    frame_number += 1
    
    # Пример обработки
    if ret:
        
        # Update road detector
        seg_map, filtered_boxes, filtered_scores = roadEstimator(frame)
        # print('new_frame: ', new_frame.shape)
        # print('seg_map: ', seg_map.shape)

        mask = get_mask_arr(seg_map, frame)
        mask_b = get_mask_arr_b(mask)
        # np.save('numpy_mask_b.npy',mask_b)
        
        # print('mask: ', mask.shape)
        if DEBUG:
            cv2.imshow("Road Detections", mask)

        
        out.write(f"{INPUT_FILE}_{frame_number},")
        out.write(mask2rle(mask_b))
        print(f"save image CSV: {INPUT_FILE}_{frame_number}")
        out.write('\n')
        
        # Остановить через 10 кадров
        # if frame_number > 10:
        #     break
    else:
        break

# Завершить запись и чтение с файлов
cap.release()

out.close()
