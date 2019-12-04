import cv2
import numpy as np
from os import path

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
''' 
factor_precision_t = [150,255] 
ret , thershold = cv2.THRESH_BINARY(rendery, factor_precision_t[0] , factor_precision_t[1],cv2.THRESH_BINARY)
'''
#fin de los algoritmos de abstracción de imagenes

#Busqueda de contornos
contours_c , herarchy_c = cv2.findContours(
            binary_c ,
            cv2.RETR_TREE , 
            cv2.CHAIN_APPROX_SIMPLE
    )   

for _contour in contours_c
    cv2.drawContours(img, c , 0 , (0,0,255) , 2)
    cv2.imshow('busqueda por canny', img)
    cv2.waitKey(0)

''' 
contours_t , herarchy_t = cv2.findContours(
            binary_t ,
            cv2.RETR_TREE , 
            cv2.CHAIN_APPROX_SIMPLE
    )   
'''
