import cv2

class FeatureExtractor ():
    def __init__ (self, model):
        # load our serialized face embedding model from disk
        self.embedder = cv2.dnn.readNetFromTorch(model)
        self.embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        #TARGET_MYRIAD

    def run (self, face):
        try:
            # construct a blob for the face ROI, then pass the blob
            # to face embedding model to obtain the 128-d dimension vector
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            self.embedder.setInput(faceBlob)
            vec = self.embedder.forward()
            return vec.flatten()

        except Exception as e:
            raise e