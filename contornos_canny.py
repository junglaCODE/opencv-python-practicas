import cv2
import numpy as np

def saveAbstraccionObjects(name , data) : 
    _data = ' '.join([str(elem) for elem in data]) 
    archivo = open('abstracciones/'+name+'.dat', 'w+')
    archivo.write(_data)
    archivo.close() 

_img = 'silla'
_ext = '.jpg'
_resource = 'assets/'+_img+_ext

img = cv2.imread(_resource)
render = cv2.cvtColor(cv2.imread(_resource),cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(render,50,500)
canny = cv2.dilate(canny, None, iterations=5)
canny = cv2.erode(canny, None, iterations=1)

contours , hirarchy = cv2.findContours(canny,
        cv2.RETR_EXTERNAL , 
        cv2.CHAIN_APPROX_SIMPLE
    )

#saveAbstraccionObjects(_img+'_h',hirarchy)
#saveAbstraccionObjects(_img+'_c',contours)

cv2.drawContours(img,contours, -1 , (0,255,0) , 2)
#cv2.imshow('canny', np.hstack([render , canny]))
cv2.imshow('buscador contornos', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

