import cv2
import numpy as np

Ip_Cam =  { 
        'left'    :   '192.168.100.20', 
        'middle'  :   '192.168.100.17', 
        'right'   :   '192.168.100.19'
     }


Streaming = cv2.VideoCapture('rtsp://'+Ip_Cam['middle']+'/stream1')

if !Streaming.isOpened() :
    print('Camara no conectada')
    return -1

while True:
  _ret,_frames= Streaming.read()

  if _ret==True:
    cv2.imshow('Video de la camara', _frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break

Streaming.release()
