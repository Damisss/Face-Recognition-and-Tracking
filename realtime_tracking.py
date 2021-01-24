#Usage
# python realtime_tracking.py \
# --input optional \
# --confidence optional (default is .9) \
# --cosine_distance optional (default is .9) \
# --nms_threshold optional (default .8) \
# --output optional (default output/result.avi) \

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import numpy as np
from multiprocessing import Process, Queue, Value 
import cv2
import pickle
import datetime
import argparse

from utils.face_detector import FaceDetector
from utils.pedestrian_detector import PedestrianDetector
from utils.embedding_extractor import FeatureExtractor
from utils.video_writer import VideoWriter
from utils import config, helper


#deep sort
from deep_sort import generate_detections as gdet
from deep_sort import nn_matching 
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', help='Path to input video.')
ap.add_argument('-c', '--confidence', default=.5, type=float,  help='Minimum proba to filter weak detections')
ap.add_argument('-d', '--cosine_distance', default=.9, type=float,  help='Cosine distance')
ap.add_argument('-t', '--nms_threshold', default=.8, type=float,  help='Non-maximum suppression threshold')
ap.add_argument('-o', '--output', default='output/result.avi',  help='Path to output video')
args = vars(ap.parse_args())

class RecognitonAndTracking ():
  @staticmethod
  def start ():
    try:
      
      # ssdmobilenet v2 labels 
      LABELS = config.LABELS
      # face recognition labels
      le = pickle.loads(open(config.LABEL_PATH, 'rb').read())
      #classes to be detected
      LIST_CLASSES = 'person'.split(',')
      # recognizer label (svm model)
      recognizer = pickle.loads(open(config.MODEL_PATH, 'rb').read())
      # load face features exractor from disk
      embedder = FeatureExtractor(config.EMBEDDING_MODEL_PATH)
      # load face recognizer from disk
      faceDetector = FaceDetector(config.FACE_DECTOR_PATH)
      # load pretrained ssdmobilenet from disk
      pedestrianDetector = PedestrianDetector(config.SSD_MOBILENET_V2_PATH, faceDetector.detector.device_path())
      
      # trackers
      #load deepsort model from disk
      encoder = gdet.create_box_encoder(config.TRACKER_MODEL_PATH, batch_size=1)
      #instatiate metrics
      metric = nn_matching.NearestNeighborDistanceMetric('cosine', args['cosine_distance'], None)
      # instatiate tracker
      tracker = Tracker(metric)
      
      if not args.get('input', False):
        cap = cv2.VideoCapture(1)
      else:
        cap = cv2.VideoCapture(args['input'])
        
      # instatiate video writer 
      videoWriter = VideoWriter(args['output'], cap.get(cv2.CAP_PROP_FPS))
      
      # let's declare some variable 
      H, W = None, None
      # objects dict holds different ids and their corresponding class name
      objects={}
      # holds the tracked objects and the number of consecutive frames they have been marked as disappeared
      disappearedIds = {}
      # total frame counts will be used to compute average fps
      frameCounts = 0
      # frames determine number of frame  allows for an object to be disappeared 
      #frames = 0
      # track object id
      trackedOjects = [] 
      # video writer process
      process = None
      # inference start time
      startTime = datetime.datetime.now()
      
      while cap.isOpened():
        grabbed, frame = cap.read()
        # stop while loop if is there is no frame available 
        if not grabbed:
          break
        #stop while loop if q is pressed
        key = cv2.waitKey(1) 
        if key == ord('q'):
          break
        
        if H is None or W is None:
          H, W = frame.shape[:2]
        
        # start video writer process
        if args['output'] and process is None:
          writeVideo = Value('i', 1)
          frameQueue = Queue()
          process = Process(target=videoWriter.start, args=(writeVideo, frameQueue, H, W))
          process.start()
        if frameCounts % 3 == 0: 
          results = pedestrianDetector.detect(frame, args['confidence'])
        bboxes =[]
        scores = []
        #frames += 1
        #loop over detection results
        for r in results:
          # label index
          label_id = int(r.label_id + 1)
          # check if detected object is in the list of the object we are interrested in.
          if label_id in LABELS.keys() and LABELS[label_id] in LIST_CLASSES:
            # scale bounding box
            bbox = r.bounding_box.flatten() * np.array([W, H, W, H])
            bbox= bbox.astype('int')
            # add object bounding box to bboxes
            bboxes.append(bbox)
              # add detection confidence to scores list
            scores.append(r.score)
        
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        if frameCounts % 3 == 0: 
          #grab feartures from frame
          features = encoder(frame, bboxes)
          # perform deep sort detection
          detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]
          # grab boxes, scores, and classes_name from deep sort detections
          pedestrianBoxes = np.array([d.tlwh for d in detections])
          pedestrianScores = np.array([d.confidence for d in detections])
          #perform non-maxima suppression on deep sort detections
          indexes = preprocessing.non_max_suppression(pedestrianBoxes, args['nms_threshold'], pedestrianScores)
          detections = [detections[i] for i in indexes]
          # update tracker
          tracker.predict()
          tracker.update(detections)
        # loop over tracked objects
        
        for track in tracker.tracks:
          if not track.is_confirmed() or track.time_since_update > 1:
            continue
          
          pedestrianBox = track.to_tlbr()
          startX, startY, endX, endY = int(pedestrianBox[0]), int(pedestrianBox[1]), int(pedestrianBox[2]), int(pedestrianBox[3])
          trackedOjects.append(track.track_id)
          
          # Perfrom face detection only if there is no object in objects dict or current track id is not objects keys
          if len(objects) == 0 or  track.track_id not in objects.keys():
            faces = faceDetector.detect(frame, .3)
            # loop over detected faces 
            for face in faces:
              bbox = face.bounding_box.flatten() * np.array([W, H, W, H])
              xmin, ymin, xmax, ymax = bbox.astype('int')
              # check if detexcted face bboxe is inside current pedestrian bbox
              isOverlap = helper.overlap((startX, startY, endX, endY), (xmin, ymin, xmax, ymax))
              # 
              if isOverlap:
                # grab ROI (face in our case) and then pass it to face feature extractor
                roi = frame[ymin:ymax, xmin:xmax]
                vector = embedder.run(roi)
                # preprocess the feature vector extracted
                vector = np.expand_dims(vector, axis=0)
                # pass the preprocessed vector to perform prediction us trained ML model (SVM in our case)
                proba = recognizer.predict_proba(vector)
                ind = np.argmax(proba[0])
                # filter out weak face detection
                if proba[0][ind] > .6:
                  label = le.classes_[ind]
                  # perform list comprehension to avoid two different ids using same name.
                  if label in objects.values():
                  #and track.track_id not in objects.keys():
                    #del objects[track.track_id]
                    objects = {k:v for k, v in objects.items() if v != label}
                    continue
                  
                  if label in objects.values() and track.track_id in objects.keys():
                    helper.drawBox(frame, (startX, startY, endX, endY), label, (0, 255, 0))
                  
                  else:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    helper.drawBox(frame, (startX, startY, endX, endY), label, (0, 255, 0))
                 
                  # add ids to objects
                  objects[track.track_id] = label 
                  # set number of disappear for current track id  to 0 
                  disappearedIds[track.track_id] = 0
    
          else: 
            # tracked already recognized people
            for key, name in objects.items():
              if key == track.track_id:
                disappearedIds[track.track_id] = 0
                helper.drawBox(frame, (startX, startY, endX, endY), name, (0, 255, 0))
        # copy  disappearedIds dict and increase the number of disappeared for a specific tracked object     
        idsCopy = disappearedIds.copy()    
        for i in idsCopy: 
          if i not in trackedOjects:
            disappearedIds[i] += 1
          #delete a given object if its number of dissapeared reached the maximum.
          # 300 is the number of maximum consecutive frame a given tracked id (object) is allowed to be marked as disappeared
          #until it is deleted from tracking.
          if  disappearedIds[i] == 300:
            del disappearedIds[i]
            if i in objects.keys():
              del objects[i]
            print(objects)
        
        trackedOjects = []
        
              
                  
        font = cv2.FONT_HERSHEY_SIMPLEX 
           
        elips= (datetime.datetime.now()-startTime).total_seconds()
        frameCounts += 1 
        fps = frameCounts/elips
        text = 'Average FPS: {:.2f}'.format(fps)
        cv2.putText(frame, text, (10, 20), font, 0.5, (0, 255, 0), 2)
        # put frame in queue
        if process is not None: 
          frameQueue.put(frame)
          
        cv2.imshow('Face Recognition and Pedestrian Tracking', frame)
      # stop writting process
      if process is not None:
        writeVideo.value = 0
        process.join()
          
      cap.release()
      cv2.destroyAllWindows()
    
    except Exception as e:
      raise e 
    
    

if __name__ == '__main__':
  RecognitonAndTracking.start()
  