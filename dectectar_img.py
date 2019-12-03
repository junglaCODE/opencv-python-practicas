import cv2
import os.path
from os import path
import numpy as np
import imutils

#configurando
filtro = { 'alpha' : -1 , 'gray': 0 , 'normal' : 1 }
resource = 'assets/oficina.jpg'

# establecimiento de exepciones
if not path.exists(resource) :
    print("recurso no encontrado")
    exit(0);

#leemos la imagen
render = cv2.imread(resource,filtro['gray'])
#parametro para hacer un mejor umbral
render_blur = cv2.medianBlur(render,5)

# algoritmos identificadores
_variable = [150,255]
ret1, binary = cv2.threshold(render, _variable[0] , _variable[1], cv2.THRESH_BINARY)
ret2, binary_inv = cv2.threshold(render, _variable[0] , _variable[1], cv2.THRESH_BINARY_INV)
ret3, zero = cv2.threshold(render, _variable[0] , _variable[1], cv2.THRESH_TOZERO)
adaptative_mean = cv2.adaptiveThreshold(render,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
adaptative_thresh = cv2.adaptiveThreshold(render_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,11)


# fin de los algoritmos identificadores

cv2.imshow('Tipos threshold binary' , 
        np.hstack([binary , binary_inv])
    )
cv2.imshow('Tipos adaptative threshold' ,
    np.hstack([ adaptative_mean , adaptative_thresh])
)
cv2.waitKey(0)
cv2.destroyAllWindows()

