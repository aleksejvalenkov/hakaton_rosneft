#!/usr/bin/python3

import cv2
from utils import mask2rle
import time

# Файлы ввода-вывода
INPUT_FILE = 'input.webm'
OUTPUT_FILE = "output.csv"

# Считать видео
cap = cv2.VideoCapture(INPUT_FILE)
frame_number = 0

if cap.isOpened() == False:
    print("Error opening video file")

# Подготовить запись файла
out = open(OUTPUT_FILE, "w")
out.write("ImageID,EncodedPixels\n")



# Основной цикл
while True:
    ret, frame = cap.read()
    frame_number += 1
    
    # Пример обработки
    if ret:
        th, frameGray = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        
        out.write(f"{INPUT_FILE}_{frame_number},")
        out.write(mask2rle(frameGray))
        out.write('\n')
        
        # Остановить через 10 кадров
        if frame_number > 10:
            break
    else:
        break

# Завершить запись и чтение с файлов
cap.release()

out.close()
time.sleep(1000)
