import time
import cv2

# Verwende die integrierte Haar Cascade XML-Datei für die Erkennung von Augen.
eyeCascadeClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_objects(image, objectClassifier):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = objectClassifier.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return objects

def draw(photo):
    image = photo.copy()
    eyes = detect_objects(image, eyeCascadeClassifier)
    
    # Zeichne Rechtecke um beide erkannten Augen.
    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Image', image)
    key = cv2.waitKey(10)
    
    return key

def main():
    camera = cv2.VideoCapture(0)
    while True:
        success, photo = camera.read()
        key = draw(photo)
        if key > 0:
            break

if __name__ == '__main__':
    main()
