import cv2
import numpy as np

MIN_POINT = 35  

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface")

class VideoCamera(object):
    img_source = []
    temp_kp_desc = []

    def __init__(self):
        
        cv2.useOptimized()
        self.video = cv2.VideoCapture(0)

        detector = cv2.xfeatures2d.SIFT_create()
        
        #Preload Data Train
        for i in range(28):
            self.img_source.append(cv2.imread('SRC/'+str(i+1)+'.bmp', 0))
            temp_kp, temp_desc = detector.detectAndCompute(self.img_source[i],None)
            self.temp_kp_desc.append([temp_kp, temp_desc])
    
    def __del__(self): #waktu servernya ditutup camera juga tutup
        self.video.release()
    
    def get_frame(self):
        return self.img_source, self.temp_kp_desc, self.video

        