# utils.py
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import cv2

IMAGE_SIZE = (128, 128)
detector = MTCNN()

def extract_face(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img)
    if not faces:
        return None

    x1, y1, w, h = faces[0]['box']
    x2, y2 = x1 + w, y1 + h
    face = img[y1:y2, x1:x2]

    face = Image.fromarray(face)
    face = face.resize(IMAGE_SIZE)
    face = np.asarray(face) / 255.0  # Normalize

    return np.expand_dims(face, axis=0)  # Shape (1, 128, 128, 3)
