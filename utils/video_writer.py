import cv2

class VideoWriter ():
  def __init__ (self, path, fps):
    self.fps = fps
    self.path = path
  
  def start (self, writeVideo, frameQueue, H, W):
    try:
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      writer = cv2.VideoWriter(self.path, fourcc, self.fps,( W, H), True)
      
      while  writeVideo.value or not frameQueue.empty():
        if not frameQueue.empty():
          frame = frameQueue.get()
          writer.write(frame)
          
      writer.release()
    except Exception as e:
      raise e