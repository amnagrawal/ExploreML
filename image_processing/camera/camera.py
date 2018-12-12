import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, img = cap.read()
    cv2.imshow('output', img)

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale', img2)

    #BGR to binary (RED) threshold image
    imgthreshold = cv2.inRange(img, (0, 0, 0), (30, 30, 30))
    cv2.imshow('thresholded', imgthreshold)
    k = cv2.waitKey(10)
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()