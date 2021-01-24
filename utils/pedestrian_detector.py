from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import cv2

class PedestrianDetector ():
  def __init__(self, modelPath, device_path):
    self.detector = DetectionEngine(modelPath, device_path)
    
  def detect (self, image, confidence):
    try:
      img = cv2.resize(image, (300, 300))
      img = Image.fromarray(image)
      results = self.detector.detect_with_image(img, threshold=confidence)
      return results
    except Exception as e:
      raise e 