import cv2
import os.path
from os import path

#configurando
filtro = { 'alpha' : -1 , 'gray': 0 , 'normal' : 1 }
resource = 'assets/oficina.jpg'

# establecimiento de exepciones
if not path.exists(resource) :
    print("recurso no encontrado")
    exit(0);

render = cv2.imread(resource,filtro['normal'])
# algoritmos identificadores

# fin de los algoritmos identificadores
cv2.imshow('Detectar con imagenes',render)

cv2.waitKey(0)
cv2.destroyAllWindows()