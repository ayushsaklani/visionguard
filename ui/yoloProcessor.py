from ultralytics import YOLO
import numpy as np
from threading import Thread
try:
    # Works when we're at the top lovel and we call main.py
    from ..trainer.model import VisionGuard
    from ..trainer.utils import *
except ImportError:
    # If we're not in the top level
    # And we're trying to call the file directly
    import sys
    # add the submodules to $PATH
    # sys.path[0] is the current file's path
    sys.path.append(sys.path[0] + '/..')
    from trainer.model import VisionGuard
    from trainer.utils import *

class YOLOProcessor:
    def __init__(self, weights='yolov8n.pt',in_q = None,out_q=None):

        self.model = YOLO(weights)
        self.model.fuse()
        self.stopped = False
        self.in_q = in_q
        self.out_q = out_q
        self.frame_count = 0
        self.device = get_device()
    
    def start(self):
        Thread(target=self.process, args=()).start()
        return self
    def stop(self):
        self.stopped = True
        
    def process(self):
        while True:
            if self.stopped:
                return
            if not self.in_q.empty(): 
                frame = self.in_q.get()
                if self.frame_count%10 == 0:
                    out = self.model.predict(frame,verbose=False)
                    boxes = out[0].boxes
                    classes = boxes.cls.detach().cpu().numpy()
                    conf = boxes.conf.detach().cpu().numpy()
                    person = np.where((conf >=0.5) & (classes ==0))[0]
                    self.bboxes = boxes.data.detach().cpu().numpy()[person]
                self.frame_count +=1
                self.out_q.put((frame,self.bboxes))
               
    