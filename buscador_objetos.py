import cv2
import numpy as np
from os import path

# funciones para ahorro de trabajo

def visualizarMomentosArea(contours) :
    for _c in contours : 
        cv2.drawContours(img, [_c] , 0 , (0,0,255) , 2)
        cv2.imshow('busqueda por canny', img)
        cv2.waitKey(0)


_resource = "assets/silla.jpg"

if not path.exists(_resource) :
    print("recurso no encontrado")
    exit(0)

# recuperacion de imagenes en diferentes formatos para se procesadas
img = cv2.imread(_resource)
render = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Algoritmos para abstraccion de imagen

#https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html


factor_precision_c = [50,500]
canny = cv2.Canny(render , factor_precision_c[0] , factor_precision_c[1] )

#https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold

factor_precision_t = [150,255] 
ret , thershold = cv2.threshold(render , factor_precision_t[0] , factor_precision_t[1],cv2.THRESH_BINARY)

#fin de los algoritmos de abstracción de imagenes

#Busqueda de contornos
contours , herarchy = cv2.findContours(
            canny ,
            cv2.RETR_EXTERNAL , 
            cv2.CHAIN_APPROX_SIMPLE
    )   

for _c in contours :
    epsilon = 0.01*cv2.arcLength(_c,True)
    approx = cv2.approxPolyDP(_c,epsilon,True)
    
    if approx >= 14 :
        cv2.putText(image,'Silla', (x,y-5),1,1,(0,255,0),1)
        x,y,w,h = cv2.boundingRect(approx)
        cv2.drawContours(img , [approx], 0, (0,255,0),2)

# Representacion Salida
cv2.imshow('adivina quien es ? ',img)
cv2.waitKey(0)
cv2.destroyAllWindows()