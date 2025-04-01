import cv2 as cv
import numpy as np

img = cv.imread(r"a\fondo.jpg", 1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

ub=np.array([0,120,40])
ua=np.array([10,255,255])

ub1=np.array([170, 40,40])
ua1=np.array([180,255,255])

mask1 = cv.inRange(hsv, ub, ua)
mask2 = cv.inRange(hsv, ub1, ua1)

mask = mask1 + mask2

res = cv.bitwise_and(img, img, mask=mask)

cv.imshow('res', res)
cv.imshow('hsv', hsv)
cv.imshow('mask', mask)

cv.waitKey(0)
cv.destroyAllWindows()