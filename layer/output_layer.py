import cv2 as cv
import os

model = "Luis"
ruta_banck = "C:/Users/llore/Cursos MISD/Vision_artificial/Deteccion/train/images_banck"

ruta = ruta_banck+ "/" + model

if not os.path.exists(ruta):
    os.makedirs(ruta)

cam = cv.VideoCapture(1)

ruidos = cv.CascadeClassifier("C:/Users/llore/Cursos MISD/Vision_artificial/Deteccion/train/hascarcades/haarcascade_frontalface_default.xml")
print(ruidos)
id = 1

while True:
    respuesta, capture = cam.read()
    if respuesta == False:break

    gray = cv.cvtColor(capture, cv.COLOR_BGR2GRAY)
    idCapture = capture.copy()
    face = ruidos.detectMultiScale(gray, 1.3, 5)

    for(x, y, e1, e2) in face:
        cv.rectangle(capture, (x, y), (x + e1, y + e2), (0,0,255), 1)
        faceCapture = idCapture[y:y+e1, x:x+e2]
        faceCapture = cv.resize(faceCapture, (160,160), interpolation= cv.INTER_CUBIC)
        cv.imwrite(ruta + "/Image_{}.jpg".format(id), faceCapture)
        id = id+1

    cv.imshow("Faces", capture)

    if id == 100:
        break
cam.release()
cv.destroyAllWindows()