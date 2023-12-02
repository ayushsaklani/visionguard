import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from ultralytics import YOLO
import numpy as np
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
    def __init__(self, model_path='/Users/shunya/Project/visionguard/ui/weights/swin_transformer_combined.pth'):


        self.device = "cpu"
        self.model = VisionGuard()
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        
        self.augument = get_augumentations()
        self.normz = get_normalize_t()
        self.print_attr = False 
        self.device = get_device()
        self.model.to(self.device)

    def __call__(self,input_x):
        input_x = Image.fromarray(input_x, 'RGB')
        input_x = self.normz(self.augument(input_x)).unsqueeze(0).to(self.device)
        attr, _ = self.model(input_x)
        attr = torch.sigmoid(attr).detach().cpu().numpy()
        attr[attr>=0.5] =1
        attr[attr<0.5] = 0
        classes = np.where(attr==1)[1]
        labels = [self.model.c2l(c) for c in classes]
        return labels
    

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("455x720")
        self.width = 405
        self.height = 720
        self.video_source = video_source
        self.frame_count =0
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window,width=self.width,height = self.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_print_attr = tkinter.Button(window, text="AttributeDetection", width=50, command=self.print_attr)
        self.btn_print_attr.pack(anchor=tkinter.CENTER,expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.yolo = YOLO("yolov5nu.pt")
        self.bboxes = None
        self.visionguard =VisionGuardProcessor()
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def print_attr(self):
        self.print_attr = not self.print_attr

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        
        frame = cv2.resize(frame,(self.width,self.height))
        
        if self.print_attr and self.frame_count%5==0:
            out = self.yolo(frame)
            boxes = out[0].boxes
            classes = boxes.cls.detach().cpu().numpy()
            person = np.where(classes ==0)[0]
            
            self.bboxes = boxes.data.detach().cpu().numpy()[person]

        if self.print_attr:
            for box in self.bboxes:
                x,y,w,h,_,_ = box
                x,y,w,h = int(x),int(y),int(w),int(h)
                frame_vis = frame[y:y+h,x:x+w]
                cv2.imwrite("lolhaha.png", frame_vis)
                labels = self.visionguard(frame_vis)
                cv2.rectangle(frame, (x, y), ( w,   h), (36,255,12), 2)
                for label in labels:
                    y = y+30
                    frame = cv2.putText(frame, label, (x+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,10,10), 1)



        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.frame_count +=1
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
video_path = "/Users/shunya/Project/visionguard/ui/videos/IMG_0311.MOV"
App(tkinter.Tk(), "Tkinter and OpenCV",video_path)