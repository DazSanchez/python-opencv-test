import os
import cv2


def _get_internal_cascade(name):
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))


def _get_external_cascade(name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cascades_dir = os.path.join(BASE_DIR, 'data', 'cascades')
    return cv2.CascadeClassifier(os.path.join(cascades_dir, name))


def get_cascades():
    face_cascade = _get_internal_cascade('haarcascade_frontalface_alt.xml')
    eyes_cascade = _get_internal_cascade('haarcascade_eye.xml')
    smile_cascade = _get_internal_cascade('haarcascade_smile.xml')
    nose_cascade = _get_external_cascade('Nose.xml')
    mouth_cascade = _get_external_cascade('Mouth.xml')

    return (face_cascade, eyes_cascade, nose_cascade, mouth_cascade, smile_cascade)
