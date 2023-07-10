import os
from ultralytics import YOLO
import cv2
import numpy as np

#video_path = "C:\\Users\\PMTC-ELE\\Desktop\\video_sop\\20230623_102359.mp4"
#video_path_out = '{}_out.mp4'.format(video_path)

model_path = "C:\\Users\\PMTC-ELE\\Desktop\\video_sop\\last.pt"

class_name_dict = {  0 : "QR code scanning",
                   1 : "spindle screw driver",
  2 : "spindle screw passenger",
  3 : "Go/No Go RHS",
  4 : "Go/No Go LHS"}

threshold = 0.5
detected_objects=[]

def video_detection(path_x):
    
    cap = cv2.VideoCapture(path_x)
    
   
    
    model = YOLO(model_path)  # load a custom model
    class_list=[]

    while True:
        ret, img = cap.read()
        results = model(img,stream =True)
         H, W, _ = img.shape
        
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            class_list.append(class_id)

            if score > threshold:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(img, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                detected_objects.append(class_name_dict[int(class_id)].upper())

    return detected_objects 




cv2.destroyAllWindows()




video_detection(path_x=0)

