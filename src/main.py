import cv2
import os
from cascades import get_cascades


def _get_image(name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, 'input')
    img_path = os.path.join(image_dir, name)

    return cv2.imread(img_path)


def detect(img):
    (face_cascade, eyes_cascade, nose_cascade, mouth_cascade, smile_cascade) = get_cascades()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 10)

    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_color, 1.2, 14)
        nose = nose_cascade.detectMultiScale(roi_color, 1.15, 10)
        mouth = mouth_cascade.detectMultiScale(roi_color, 1.35, 30)
        smile = smile_cascade.detectMultiScale(roi_color, 1.5, 30)
        # cv2.imwrite(img_path.replace("input","output"), roi_color)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny),
                          (nx+nw, ny+nh), (255, 255, 255), 2)
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
        for (mx, my, mw, mh) in smile:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 0, 255), 2)

img = _get_image('face4.jpg')
detect(img)

cv2.imshow('face', img)

cv2.waitKey(0)
