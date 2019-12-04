import cv2

print('tipo de efecto');
filtro = int(input())
imagen = cv2.imread('assets/example_1.jpg',filtro);
cv2.imshow("Render Imagen",imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()