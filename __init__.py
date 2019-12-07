import cv2
import numpy as np
from os import path

Ip_Cam =  { 
        'left'    :   '192.168.100.20', 
        'middle'  :   '192.168.100.17', 
        'right'   :   '192.168.100.19',
	'local'   : 0	
     }

# Configurando entorno
Streaming = cv2.VideoCapture(0)
cv2.namedWindow('Binarizacion', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camara', cv2.WINDOW_NORMAL)

# Capaturando Exepciones de trasmision de video
if not Streaming.isOpened() :
    print('Camara no conectada')
    exit(0)

# Generando el Streaming capturado
while Streaming.isOpened():
    _ret,_frames= Streaming.read()
    render = cv2.cvtColor(_frames,cv2.COLOR_BGR2GRAY)

    factor_precision = [50,500]
    render = cv2.Canny(render , factor_precision[0] , factor_precision[1] )
    render = cv2.dilate(render, None, iterations=15)
    render = cv2.erode(render, None, iterations=1)

    contours , herarchy = cv2.findContours(
            render ,
            cv2.RETR_EXTERNAL , 
            cv2.CHAIN_APPROX_SIMPLE
    )   

    archivo = open('objetos.dat', 'w+')
    _data = ''

    for _c in contours :

        area = cv2.contourArea(_c)

        if area > 0 :

            epsilon = 0.01*cv2.arcLength(_c,True)
            approx = cv2.approxPolyDP(_c,epsilon,True)    
            x,y,w,h = cv2.boundingRect(approx)
            '''
            if len(approx) <= 4 :
                cv2.putText(_frames,'algo cuadrado', (x,y-5),1,1,(0,255,0),1)

            if len(approx) >= 5 and len(approx) < 7 :
                cv2.putText(_frames,'boligrafo', (x,y-5),1,1,(0,255,0),1)
                _data = 'Boligrafo'

            if len(approx) >= 7 and len(approx) < 12 :
                cv2.putText(_frames,'Taza', (x,y-5),1,1,(0,255,0),1)
                _data = 'Taza'

            if len(approx) > 12 and len(approx) < 26 :
                cv2.putText(_frames,'Silla', (x,y-5),1,1,(0,255,0),1)
                _data = 'Silla'
            '''
            if len(approx) > 25 :
                cv2.putText(_frames,'Una persona ?', (x,y-5),1,1,(0,255,0),1)

            cv2.drawContours(_frames , [approx], 0, (0,255,0),2)
            archivo.write(_data + '\n')

        if _ret==True:
            cv2.imshow('Binarizacion', render)
            cv2.imshow('Camara', _frames)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

archivo.close() 
Streaming.release()
cv2.destroyAllWindows()
