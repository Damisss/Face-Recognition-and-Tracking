import pickle
import cv2
import os
import numpy as np
from utils.data_loader import DataLoader
from utils import config
from utils.embedding_extractor import FeatureExtractor
from utils.face_detector import FaceDetector

def prepareData (path):
    try:
        # embedder
        embedder = FeatureExtractor(config.EMBEDDING_MODEL_PATH)
        # face detector
        faceDetector = FaceDetector(config.FACE_DECTOR_PATH)
        # image paths
        imagePaths = DataLoader(config.DATASET_PATH)
        names = []
        embeddedVectors = []
        for imagePath in imagePaths:
            name = imagePath.split(os.path.sep)[1]
            img = cv2.imread(imagePath)
            H, W = img.shape[:2]
            # face detection
            detectedFaces = faceDetector.detect(img, .3)
            for detectedFace in detectedFaces:
                #grab bounding box
                bbox = detectedFace.bounding_box.flatten() * np.array([W, H, W, H])
                xmin, ymin, xmax, ymax = bbox.astype('int')
                # grab ROI from image
                roi = img[ymin:ymax, xmin:xmax]
                vector = embedder.run(roi)
                embeddedVectors.append(vector)
                names.append(name)

        data = {'data':embeddedVectors, 'names': names}
    
        with open(path, 'wb') as f:
            f.write(pickle.dumps(data))

    except Exception as e:
        raise e