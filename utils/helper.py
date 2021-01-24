import cv2

def overlap (boxA, boxB):
  try:
 
    if(boxA[0] < boxB[0] and boxA[0] + boxA[2] > boxB[0] + boxB[2] and boxA[1] + boxA[3] > boxB[1] + boxB[3]):
      return True
    
    if (boxA[0] - boxB[0]/3 < boxB[0]  and boxA[0] + boxA[2] > boxB[0] + boxB[2] and boxA[1] + boxA[3] > boxB[1] + boxB[3]):
      return True
    
    if (boxA[2] - boxB[2]/3 < boxB[2]  and boxA[0] + boxA[2] < boxB[0] + boxB[2] and boxA[1] + boxA[3] > boxB[1] + boxB[3]):
      return True

  except Exception as e:
    raise e
  
  

def drawBox (frame, box, label, color):
  try:
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = box[1] - 30 if box[1] - 30 > 30 else box[1] + 30
    t = box[1] - 15 if box[1] - 30 > 30 else box[1] + 15
    cv2.rectangle(frame, (box[0], y), ((box[0]+len(label)* 15), box[1]), color, -1)
    cv2.putText(frame, label.capitalize(), (box[0], t), font, 0.5, (255, 255, 255), 2)
  except Exception as e:
    raise e
  