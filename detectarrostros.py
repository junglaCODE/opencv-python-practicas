import cv2
import numpy as np

_red_neuronal = 'redes_neuronales/haarcascade_frontalface_default.xml'
faceClassif = cv2.CascadeClassifier(_red_neuronal)
cv2.namedWindow("Detectar Rostros", cv2.WINDOW_NORMAL)  
'''
#imagen estatica
image = cv2.imread('assets/oficina.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=11,
	minSize=(50,50),
	maxSize=(400,400)
)

for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)


cv2.imshow('Detectar Rostros',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#video
'''
cap = cv2.VideoCapture(0);

while True:
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#render = cv2.Canny(gray, 100, 200 )

	faces = faceClassif.detectMultiScale( 
		gray ,
		scaleFactor=1.1,
		minNeighbors=11,
		minSize=(50,50),
		maxSize=(400,400)
	)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow('Detectar Rostros',frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break