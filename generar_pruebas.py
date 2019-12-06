import cv2
import numpy as np
from os import path

Ip_Cam =  { 
        'left'    :   '192.168.100.20', 
        'middle'  :   '192.168.100.17', 
        'right'   :   '192.168.100.19'
     }

# Configurando entorno
Streaming = cv2.VideoCapture('rtsp://'+Ip_Cam['middle']+'/stream1')

cascPath = "redes_neuronales/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

count = 0

while  Streaming.isOpened() :

    _ret , _frames = Streaming.read()

    negative = cv2.cvtColor(_frames, cv2.COLOR_BGR2GRAY)

    rostro = faceCascade.detectMultiScale(negative, 1.5, 5)
    for(x,y,w,h) in rostro:
        
        x = x - 70 #suponiendo que es el casco
        y = y - 170 #suponiendo que es el casco
        w = w + 130 #suponiendo que es el casco
        h = h + 180 # suponiendo que es el casco
        
        cv2.circle(_frames,(x,y),10,(0,0,255))
        cv2.rectangle(_frames, (x,y), (x+w, y+h), (255,0,0), 4)
        count += 1
        cv2.imwrite("assets/datasets/concasco/hombre_"+str(count)+".jpg", negative[y:y+h, x:x+w])

    cv2.imshow("Capturas de imagenes", _frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    elif count >= 400:
        break
    

# Cuando todo está hecho, liberamos la captura
Streaming.release()
cv2.destroyAllWindows()