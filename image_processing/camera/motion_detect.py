import cv2
import numpy as np


def diffimg(a, b, c):
    t0 = cv2.absdiff(a, b)
    t1 = cv2.absdiff(b, c)

    t2 = cv2.bitwise_and(t0, t1)
    return t2

cap = cv2.VideoCapture(0)
t = cap.read()[1]
tp = cap.read()[1]
tpp = cap.read()[1]

t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
tp = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
tpp = cv2.cvtColor(tpp, cv2.COLOR_BGR2GRAY)


while True:

    img = diffimg(t, tp, tpp)
    cv2.imshow('motion_detect', img)

    res, img = cap.read()
    t = tp
    tp = tpp
    tpp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(10)

    if key == 27:
        cv2.destroyAllWindows()
        break

print "goodbye user"
