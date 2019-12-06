import cv2
import pickle

cascPath = "redes_neuronales/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

reconocimiento = cv2.face.LBPHFaceRecognizer_create()
reconocimiento.read("redes_neuronales/modelo_seguridad.yml")

etiquetas = {"estado" : 1 }
with open("redes_neuronales/modelo_seguridad_entrenado.pickle",'rb') as f:
    pre_etiquetas = pickle.load(f)
    etiquetas = { v:k for k,v in pre_etiquetas.items()}

web_cam = cv2.VideoCapture('rtsp://192.168.100.17/stream1')

while web_cam.isOpened() :
    # Capture el marco
    ret, marco = web_cam.read()
    grises = cv2.cvtColor(marco, cv2.COLOR_BGR2GRAY)    
    rostros = faceCascade.detectMultiScale(grises, 1.5, 5)

    # Dibujar un rectángulo alrededor de las rostros
    for (x, y, w, h) in rostros:
        #print(x,y,w,h)
        roi_gray = grises[y:y+h, x:x+w]
        roi_color = marco[y:y+h, x:x+w]
        x = x - 70 #suponiendo que es el casco
        y = y - 170 #suponiendo que es el casco
        w = w + 130 #suponiendo que es el casco
        h = h + 180 # suponiendo que es el casco
        # reconocimiento
        id_, conf = reconocimiento.predict(roi_gray)
        print(conf)
        if conf >= 50  and conf < 70:
            #print(id_)
            #print(etiquetas[id_])           
            font = cv2.FONT_HERSHEY_SIMPLEX            
            nombre = etiquetas[id_]
            color = (255,255,255)
            grosor = 2
            cv2.putText(marco, nombre, (x,y), font, 1, color, grosor, cv2.LINE_AA)
            img_item = etiquetas[id_]+".png"
            cv2.imwrite(img_item, roi_gray)
        
        cv2.rectangle(marco, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display resize del marco  
    marco_display = cv2.resize(marco, (1200, 650), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Detectando Rostros', marco_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cuando todo está hecho, liberamos la captura
web_cam.release()
cv2.destroyAllWindows()