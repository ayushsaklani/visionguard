import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from ultralytics import YOLO
import numpy as np
from threading import Thread
from videoCapture import MyVideoCapture
from yoloProcessor import YOLOProcessor
from visionGuardProcessor import VisionGuardProcessor
import queue 

import warnings
warnings.filterwarnings("ignore")




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
        self.vid_q = queue.Queue()
        self.yolo_q = queue.Queue()
        self.vis_q = queue.Queue()
        self.vid = MyVideoCapture(self.video_source,self.vid_q,self.width,self.height).start()
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window,width=self.width,height = self.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)


        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1000//30
        self.yolo_processor = YOLOProcessor(in_q=self.vid_q,out_q=self.yolo_q).start()
        

        self.visionguard =VisionGuardProcessor(in_q=self.yolo_q,out_q=self.vis_q).start()
        
        self.update()

        self.window.protocol('WM_DELETE_WINDOW', self.destroy) 
        self.window.mainloop()

    def destroy(self):
        print("Shutting down threads")
        self.vid.stop()
        self.yolo_processor.stop()
        self.visionguard.stop()
        self.window.destroy()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        
        cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    def update(self):
        # Get a frame from the video source
        # ret,frame = self.vid.read()
        if not self.vis_q.empty():
            frame = self.vis_q.get()
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.frame_count +=1
        self.window.after(self.delay, self.update)





if __name__ == "__main__":
    # Create a window and pass it to the Application object
    video_path = "/Users/shunya/Project/visionguard/ui/videos/IMG_0311.MOV"
    App(tkinter.Tk(), "Tkinter and OpenCV",video_path)