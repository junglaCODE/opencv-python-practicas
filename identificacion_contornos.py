import cv2
import os.path
from os import path
import numpy as np
import imutils

#configurando
filtro = { 'alpha' : -1 , 'gray': 0 , 'normal' : 1 }
resource = 'assets/silla.jpg'
person = 'assets/persona.jpg'
# establecimiento de exepciones
if not path.exists(resource) :
    print("recurso no encontrado")
    exit(0)

#leemos la imagen
img1 = cv2.imread(resource)
img2 = cv2.imread(person)

render1 = cv2.imread(resource,filtro['gray'])
render2 = cv2.imread(person,filtro['gray'])

#parametro para hacer un mejor umbral
render_blur = cv2.medianBlur(render1,11)

# algoritmos identificadores

# Abastre la esencia de la imagen
_variable = [150,255]
ret1, binary1 = cv2.threshold(render1, _variable[0] , _variable[1], cv2.THRESH_BINARY)
ret2, binary2 = cv2.threshold(render2, _variable[0] , _variable[1], cv2.THRESH_BINARY)
#ret2, binary_inv = cv2.threshold(render, _variable[0] , _variable[1], cv2.THRESH_BINARY_INV)
#ret3, zero = cv2.threshold(render, _variable[0] , _variable[1], cv2.THRESH_TOZERO)
#adaptative_mean = cv2.adaptiveThreshold(render,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
#adaptative_thresh = cv2.adaptiveThreshold(render_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,11)

# busca contornos 
# RETR_EXTERNAL
# RETR_LIST
# RETR_CCOMP 
# RETR_TREE.

contours1,hierarchy2 = cv2.findContours(
        binary1 , 
        cv2.RETR_TREE , 
        cv2.CHAIN_APPROX_SIMPLE
    )

contours2,hierarchy2 = cv2.findContours(
        binary2 , 
        cv2.RETR_TREE , 
        cv2.CHAIN_APPROX_SIMPLE
    )

cv2.drawContours( img1 , contours1 , -1, (255,255,0), 2)
cv2.drawContours( img2 , contours2 , -1, (255,255,0), 2)


# fin de los algoritmos identificadores

''' cv2.imshow('Tipos threshold binary' , 
       np.hstack([binary , binary_inv])
    )
cv2.imshow('Tipos adaptative threshold' ,
    np.hstack([ adaptative_mean , adaptative_thresh])
)'''

cv2.imshow('Imagenes real vs simplificada 1' , img1 ) 
cv2.imshow('Imagenes real vs simplificada 2 ' , img2 ) 

cv2.waitKey(0)
cv2.destroyAllWindows()

