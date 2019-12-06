import cv2
import os
import numpy as np
from PIL import Image
import pickle

cascPath = "redes_neuronales/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#reconocimiento con opencv
reconocimiento = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"assets/datasets")


current_id = 0
etiquetas_id = {}
y_etiquetas = []
x_entrenamiento = []

for root, dirs, archivos in os.walk(image_dir):
    for archivo in archivos:
        if archivo.endswith("png") or archivo.endswith("jpg"):
            pathImagen = os.path.join(root,archivo)
            etiqueta = os.path.basename(root).replace(" ", "-")


            if not etiqueta in etiquetas_id:                
                etiquetas_id[etiqueta] = current_id
                current_id += 1            
            id_ = etiquetas_id[etiqueta]

            pil_image = Image.open(pathImagen).convert("L")
            tamanio = (550,550)
            imagenFinal = pil_image.resize(tamanio, Image.ANTIALIAS)
            image_array = np.array(pil_image,"uint8")

            rostros = faceCascade.detectMultiScale(image_array, 1.5, 5)

            for (x,y,w,h) in rostros:
                #x = x - 70 #suponiendo que es el casco
                #y = y - 170 #suponiendo que es el casco
                #w = w + 130 #suponiendo que es el casco
                #h = h + 180 # suponiendo que es el casco
                roi = image_array[y:y+h, x:x+w]
                x_entrenamiento.append(roi)
                y_etiquetas.append(id_)


with open("redes_neuronales/modelo_seguridad_entrenado.pickle",'wb') as f:
    pickle.dump(etiquetas_id, f)

reconocimiento.train(x_entrenamiento, np.array(y_etiquetas))
reconocimiento.save("redes_neuronales/modelo_seguridad.yml")