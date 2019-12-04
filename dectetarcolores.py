import cv2
import numpy as np

captura = cv2.VideoCapture('assets/video_4.mkv')
cv2.namedWindow("Render Video", cv2.WINDOW_NORMAL)  
#cv2.namedWindow("Mascara Alfa", cv2.WINDOW_NORMAL)  
#cv2.namedWindow("Mascara Color", cv2.WINDOW_NORMAL)  
## Gama de colores HSV


def dibujar(mask,color,_area):
    contornos, hierarchy  = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        area = cv2.contourArea(c)
        if area > _area:
            M = cv2.moments(c)
            if (M["m00"]==0): M["m00"]=1
            x = int(M["m10"]/M["m00"])
            y = int(M['m01']/M['m00'])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(video, (x,y), 7, (0,0,0), -1)
            cv2.putText(video, '{},{}'.format(x,y),(x+10,y), font, .75 ,(243,7,241),1,cv2.LINE_AA)
            cv2.drawContours(video, [nuevoContorno], 0, color, 3)

greenInitial = np.array([45,45,45],np.uint8)
greenFinal = np.array([75,255,255],np.uint8)
redInitial1 = np.array([0, 100, 20], np.uint8)
redFinal1 = np.array([8, 255, 255], np.uint8)
redInitial2 = np.array([175, 100, 20], np.uint8)
redFinal2 = np.array([179, 255, 255], np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

while(captura.isOpened()):

    ret,video = captura.read()
    if ret==True:
        frameHSV = cv2.cvtColor(video,cv2.COLOR_BGR2HSV) 

        maskGreen = cv2.inRange(frameHSV,greenInitial,greenFinal)
        maskRed1 = cv2.inRange(frameHSV, redInitial1, redFinal1)
        maskRed2 = cv2.inRange(frameHSV, redInitial2, redFinal2)
        maskRed = cv2.add(maskRed1, maskRed2)

        #maskColor = cv2.add(maskRed1, maskRed2)
        #maskColor = cv2.add(maskColor,maskGreen)


        #cv2.drawContours(video, contornos, -1, (255,0,0), 3)

        #Observando los colores detectados   
        #maskBitwise = cv2.bitwise_and(video, video, mask=maskColor)  

        #cv2.imshow('Mascara Alfa',maskColor)
        #cv2.imshow('Mascara Color',maskBitwise)

    
        dibujar(maskGreen,(63,242,1),3000)
        dibujar(maskRed,(0,0,255),3000)
        cv2.imshow('Render Video',video)     

        if cv2.waitKey(130) & 0xFF == ord('s'):
            break
    else: break

captura.release()
cv2.destroyAllWindows()

