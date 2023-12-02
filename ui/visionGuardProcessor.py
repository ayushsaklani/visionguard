from threading import Thread
import numpy as np
import cv2
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


class VisionGuardProcessor:
    def __init__(self, model_path='/Users/shunya/Project/visionguard/ui/weights/swin_transformer_combined.pth',in_q = None,out_q=None):


        self.device = get_device()
        self.model = VisionGuard()
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.augument = get_augumentations()
        self.normz = get_normalize_t()
        self.print_attr = False 
        self.model.to(self.device)
        self.stopped = False
        self.in_q = in_q
        self.out_q = out_q
    
    def start(self):
        Thread(target=self.process, args=()).start()
        return self
    def stop(self):
        self.stopped = True
    
    def predict(self,input_x):
        input_x = Image.fromarray(input_x, 'RGB')
        input_x = self.normz(self.augument(input_x)).unsqueeze(0).to(self.device)
        attr, _ = self.model(input_x)
        attr = torch.sigmoid(attr).detach().cpu().numpy()
        attr[attr>=0.5] =1
        attr[attr<0.5] = 0
        classes = np.where(attr==1)[1]
        labels = [self.model.c2l(c) for c in classes]
        return labels
    
    def process(self):
        while True:
            if self.stopped:
                return
            if not self.in_q.empty(): 
                frame,  bboxes = self.in_q.get()

                for box in bboxes:
                    x,y,w,h,_,_ = box
                    x,y,w,h = int(x),int(y),int(w),int(h)
                    frame_vis = frame[y:y+h,x:x+w]
                    labels = self.predict(frame_vis)
                    cv2.rectangle(frame, (x, y), ( w,   h), (36,255,12), 2)
                    for label in labels:
                        y = y+20
                        frame = cv2.putText(frame, label, (x+10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,10,10), 1)
                self.out_q.put(frame)


               
                # return labels
    