import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    if(ret):
        gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #No poner espacios
        cv.imshow('salida', img)
        cv.imshow('gray', gris)
    k = cv.waitKey(1) & 0xFF
    if(k == 27):
        break
    #else:
     #   break
cap.release()
cv.destroyAllWindows()