from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import cv2

class FaceDetector ():
  def __init__(self, modelPath):
    self.detector = DetectionEngine(modelPath)
    
  def detect (self, image, confidence):
    try:
      img = cv2.resize(image, (320, 320))
      img = Image.fromarray(image)
      faces = self.detector.detect_with_image(img, threshold=confidence)
      return faces
    except Exception as e:
      raise e 
  