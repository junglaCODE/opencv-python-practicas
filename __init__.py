import cv2
import numpy as np

Ip_Cam =  { 
        'left'    :   '192.168.100.20', 
        'middle'  :   '192.168.100.17', 
        'right'   :   '192.168.100.19'
     }

# Configurando entorno
Streaming = cv2.VideoCapture('rtsp://'+Ip_Cam['middle']+'/stream1')
cv2.namedWindow('Test', cv2.WINDOW_NORMAL)

# Capaturando Exepciones de trasmision de video
if not Streaming.isOpened() :
    print('Camara no conectada')
    exit(0)

# Generando el Streaming capturado
while True:
    _ret,_frames= Streaming.read()

    if _ret==True:
        cv2.imshow('Test', _frames)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break


Streaming.release()
cv2.destroyAllWindows();