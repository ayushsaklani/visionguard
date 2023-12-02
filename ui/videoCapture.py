import cv2
from threading import Thread

class MyVideoCapture:
    def __init__(self, video_source=0,out_q=None,width=None,height=None):
        # Open the video source
        
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.stopped = False
        self.out_q = out_q
        self.width=width
        self.height = height

    def start(self):
        Thread(target=self.get_frame,args=()).start()
        return self
   
    def stop(self):
        self.stopped = True

    def read(self):
        return self.ret,self.frame
    
    def get_frame(self):
        while True:
            if self.stopped:
                return
            if  self.vid.isOpened():
                self.ret, self.frame = self.vid.read()
                if self.ret:
                    # Return a boolean success flag and the current frame converted to BGR
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    self.frame = cv2.resize(self.frame,(self.width,self.height))
                    self.out_q.put(self.frame)
                    # return (self.ret, self.frame)
            #     else:
            #         return (self.ret, None)
            # else:
            #     return (self.ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
