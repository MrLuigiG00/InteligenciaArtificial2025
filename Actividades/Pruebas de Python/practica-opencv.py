import cv2
import numpy as np
img = cv2.imread(r"a\fondo.jpg", cv2.IMREAD_COLOR)
imgn = np.zeros(img.shape[:2], np.uint8)

r,g,b = cv2.split(img)
print(img.shape)
imgb =cv2.merge([b, imgn, imgn])
imgg =cv2.merge([imgn, g, imgn])
imgr =cv2.merge([imgn, imgn, r])
imgnn =cv2.merge([r,g,b])
#img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img4 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Canales de colres en escala de grises
cv2.imshow('salida1', imgnn)
#cv2.imshow('salida2', imgg)
#cv2.imshow('salida3', imgr)
#cv2.imshow('salida4', imgb)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Para data sets el HSV es el mejor para modelar por que se puede obtener la pureza del color 
