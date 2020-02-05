#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:16:18 2020

@author: aman
"""

import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from imutils.video import VideoStream
import imutils
import time


from yolo_implementation import load_image_pixels
from yolo_implementation import decode_netout
from yolo_implementation import correct_yolo_boxes
from yolo_implementation import do_nms
from yolo_implementation import get_boxes
# from yolo_implementation import draw_boxes
print("#########################")
from PIL import Image
import cv2

print("opening videocam")
model = load_model('model.h5')

input_w, input_h = 416, 416

TOTAL = 0
vs = VideoStream(src=0).start()
time.sleep(1.0)


def draw_boxes(img, v_boxes, v_labels, v_scores):
    data = cv2.imread(img)
    cv2.imshow("image", data)
    
    for i in range(len(v_boxes)):
        box = v_boxes[i]
		# get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		# calculate width and height of the box
        # width, height = x2 - x1, y2 - y1
        data = cv2.rectangle(data, (y1, x1), (y2, x2), (0, 255, 0))
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        cv2.putText(data, label, (y1, x1))
    
    cv2.imshow("image", data)    

while True:
    frame = vs.read()
    filename = 'test.jpg'
    cv2.imwrite(filename, img=frame)
    
    image, image_w, image_h = load_image_pixels(filename, (input_w, input_h))
    yhat = model.predict(image)
    # summarize the shape of the list of arrays
    print([a.shape for a in yhat])
    # define the anchors
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    # define the probability threshold for detected objects
    class_threshold = 0.6
    boxes = list()
    for i in range(len(yhat)):
    	# decode the output of the network
    	boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # define the labels
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    # summarize what we found
    for i in range(len(v_boxes)):
    	print(v_labels[i], v_scores[i])
    # draw what we found
    draw_boxes(filename, v_boxes, v_labels, v_scores)
    
    key = cv2.waitKey(1) & 0xFF
    
    # if the 'q' key was pressed, break from the loop
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()