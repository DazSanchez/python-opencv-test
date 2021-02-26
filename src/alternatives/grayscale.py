import os
import cv2
from functools import reduce
# https://github.com/anjith2006/RotationalInvarianFacedetection
from face_tracker import FaceTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'input')


def get_cascade_path(name):
    return cv2.data.haarcascades + 'haarcascade_' + name + '.xml'


def process_img(path, trackers):
    img = cv2.imread(path)
    results = []

    for tracker in trackers:
        points = tracker.detect(img)
        if points is not None:
            img = tracker.draw_rectangle(img, points)
            results.append(points)
            cv2.imwrite(path.replace("input", "output"), img)


classifiers = [get_cascade_path("frontalface_default"), get_cascade_path(
    "frontalface_alt"), get_cascade_path("frontalface_alt2"), get_cascade_path("frontalface_alt_tree"), get_cascade_path("profileface")]

trackers = []
for cascade in classifiers:
    trackers.append(FaceTracker(cascade, 0.25, 1.3))

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            process_img(path, trackers)
