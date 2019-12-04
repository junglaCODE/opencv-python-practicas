import cv2

# para camara que esta por default
captura = cv2.VideoCapture('assets/video.mp4')
cv2.namedWindow("Render Video", cv2.WINDOW_NORMAL)  
#salida = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))
while(captura.isOpened()):
    ret,video = captura.read()
    if ret==True:
        cv2.resize(video, (960,600))
        cv2.imshow('Render Video',video)
        #salida.write(image)
        if cv2.waitKey(130) & 0xFF == ord('s'):
            break
    else: break

captura.release()
#salida.release()
cv2.destroyAllWindows()