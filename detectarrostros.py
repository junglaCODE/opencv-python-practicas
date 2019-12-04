import cv2
import numpy as np

faceClassif = cv2.CascadeClassifier('redes_neuronales/haarcascade_frontalface_default.xml')
cv2.namedWindow("Detectar Rostros", cv2.WINDOW_NORMAL)  

image = cv2.imread('assets/example_6.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(gray,
	scaleFactor=1.1,
	minNeighbors=11,
	minSize=(100,100),
	maxSize=(400,400)
)

for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow('Detectar Rostros',image)
cv2.waitKey(0)
cv2.destroyAllWindows()