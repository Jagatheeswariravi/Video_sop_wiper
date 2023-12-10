from flask import Flask, render_template, Response, session,redirect,request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
import cv2
import math
import requests
import time
import datetime
from ultralytics import YOLO
from flask import jsonify
import torch
from pymodbus.client import ModbusTcpClient
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
app = Flask(__name__)
app.config['SECRET_KEY'] = 'LucasTVS'
app.config['UPLOAD_FOLDER'] = 'static/files'

part_number = ['#12345','#12345','#12345','#12345']
part_name = ['Part Name','Part Name','Part Name','Part Name']
model_number = ['ABC12345','ABC12345','ABC12345','ABC12345']
planned_cycle = ['0.02s','0.02s','0.02s','0.02s']
total_count = 0
pass_count = 0
sop_violation = 0
cycle_time = ['0.04s','0.15s','0.25s']
fromdevicevariable = 0
detection_type =""
length=0


drop_down = {
    '26072107':['SW52 WIPER MOTOR AND SYSTEM ASSY','Y0M RHD','42.Sec'],
    '26072121':['SW52 WIPER MOTOR AND SYSTEM ASSY','Y0M LHD','42.Sec'],
    '26072194':['3SW60 WIPER MOTOR AND SYSTEM ASSY','YG8 RHD','18.Sec'],
    '26072205':['3SW60 WIPER MOTOR AND SYSTEM ASSY','YG9 RHD EXP','18.Sec'],
    '26072197':['3SW60 WIPER MOTOR AND SYSTEM ASSY','YG10 LHD','18.Sec'],
    '26072283':['3SW60 WIPER MOTOR AND SYSTEM ASSY','Y9T RHD','18.Sec'],
    '26072285':['3SW60 WIPER MOTOR AND SYSTEM ASSY','Y9T LHD','18.Sec'],
}
achieved_today = 0

cycle_time_analysis = {
    # 0:0,
    # 1:0,
    # 2:0,
    # 3:0,
    # 4:0,
    # 5:0,
    # 6:0,
    # 7:0,
    8:'0:00:00',
    9:'0:00:00',
    10:'0:00:00',
    11:'0:00:00',
    12:'0:00:00',
    13:'0:00:00',
    14:'0:00:00',
    15:'0:00:00',
    16:'0:00:00',
    17:'0:00:00'
    # 18:0,
    # 19:0,
    # 20:0,
    # 21:0,
    # 22:0,
    # 23:0
}


cycle_ui = {
    # 0:0,
    # 1:0,
    # 2:0,
    # 3:0,
    # 4:0,
    # 5:0,
    # 6:0,
    # 7:0,
    8:0,
    9:0,
    10:0,
    11:0,
    12:0,
    13:0,
    14:0,
    15:0,
    16:0,
    17:0
    # 18:0,
    # 19:0,
    # 20:0,
    # 21:0,
    # 22:0,
    # 23:0
}

production_count_hours = {
    # 0:0,
    # 1:0,
    # 2:0,
    # 3:0,
    # 4:0,
    # 5:0,
    # 6:0,
    # 7:0,
    8:0,
    9:0,
    10:0,
    11:0,
    12:0,
    13:0,
    14:0,
    15:0,
    16:0,
    17:0
    # 18:0,
    # 19:0,
    # 20:0,
    # 21:0,
    # 22:0,
    # 23:0
}


class_name_dict = {}

yom_flag = False
yg8_flag = False
ppm_flag = False
detected_objects = []
host =  '192.168.3.45'  
port = 502 #560 #502
client = ModbusTcpClient(host, port)


def video_detection_yom():
    global detected_objects
    video_capture = 0 #"rtsp://admin:Lucas123@192.168.1.64:554"
    global fromdevicevariable
    global yom_flag
    global class_name_dict
    global length
    
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("best.pt") # device='cpu' /'gpu' / model.to('cuda')
    #model.to('cuda')
    classNames = [
        "QR code scanning",
        "spindle screw driver",
        "spindle screw passenger",
        "Go/No Go RHS",
        "Go/No Go LHS"
    ]
    class_name_dict = {
    0: "QR code scanning",
    1: "spindle screw driver",
    2: "spindle screw passenger",
    3: "Go/No Go RHS",
    4: "Go/No Go LHS"
    }
    length = 4
    while yom_flag:
        #client.connect()
        success, img = cap.read()
        results = model(img,imgsz=192)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
               
                label = f"{class_name}{conf}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                if conf > 0.2:
                    if class_name not in detected_objects:
                        detected_objects.append(class_name)
        
        # rr = client.read_holding_registers(5300,1,unit=1)
        
        # if rr.registers == [1]:
            
        #     detected_objects=[]
        #     fromdevicevariable = 1
        #     client.write_registers(5322,1,unit=1)

       
        yield img

    cv2.destroyAllWindows()
 
def video_detection_yg8():
    global detected_objects
    global yg8_flag
    global fromdevicevariable
    global class_name_dict
    global length
    video_capture = 0 #"rtsp://admin:Lucas123@192.168.1.64:554"

    
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("yg8_best.pt") # device='cpu' /'gpu' / model.to('cuda')
    #model.to('cuda')
    classNames = ["Three pin checking",
                  "QR code scanning",
                  "Spindle screw driver"
        
    ]

    class_name_dict = {0 : "Three pin checking",
                       1 : "QR code scanning",
                       2 : "Spindle screw checking"}
    length = 2

    while yg8_flag:
        #client.connect()
        success, img = cap.read()
        results = model(img,imgsz=192)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
               
                label = f"{class_name}{conf}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                if conf > 0.6:
                    if class_name not in detected_objects:
                        detected_objects.append(class_name)

        # rr = client.read_holding_registers(5300,1,unit=1)
        
        # if rr.registers == [1]:
            
        #     detected_objects=[]
        #     fromdevicevariable = 1
        #     client.write_registers(5322,1,unit=1)

       
        yield img




video_detection_functions = {
    'SW52 WIPER MOTOR AND SYSTEM ASSY': "video_detection_yom",
    '3SW60 WIPER MOTOR AND SYSTEM ASSY': "video_detection_yg8"
}


def generate_frames_web():
    global yg8_flag
    global ppm_flag
    global yom_flag
    time.sleep(10)
    video_detection_function = detection_type

    print("generating web frames is called")
    print(f"video detection function is {video_detection_function}")


    if (video_detection_function == 'SW52 WIPER MOTOR AND SYSTEM ASSY'):
        yg8_flag = False
        ppm_flag =False
        yom_flag = True
        yolo_output = video_detection_yom()
        for detection_ in yolo_output:
            ref, buffer = cv2.imencode('.jpg', detection_)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if (video_detection_function == '3SW60 WIPER MOTOR AND SYSTEM ASSY'):
        yg8_flag = True
        ppm_flag =False
        yom_flag = False
        yolo_output = video_detection_yg8()
        for detection_ in yolo_output:
            ref, buffer = cv2.imencode('.jpg', detection_)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
   

@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('Updated web.html',detected_objects=detected_objects)

@app.route('/webapp')
def webapp():
    
    return Response(generate_frames_web(),mimetype='multipart/x-mixed-replace; boundary=frame')#rtsp://admin:cctv@123@192.168.1.64:554

    # print("webapp is called")
    
    # print(detection_type)
    # if detection_type in video_detection_functions:
    #     #video_detection_function = video_detection_functions[detection_type]
    #     return Response(generate_frames_web(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # else:
    #     return 'Invalid detection type'

@app.route('/fromdevice',methods=['GET', 'POST'])
def fromdevice():
    global fromdevicevariable
    fromdevicevariable  = 1
    return "hello"


@app.route('/fromdevicecheck',methods=['GET', 'POST'])
def fromdevicecheck():
    global fromdevicevariable
    devicestatus = fromdevicevariable
    fromdevicevariable  = 0
    return jsonify(devicestatus)


@app.route('/reset',methods=['GET', 'POST'])
def reset():
    global total_count
    global sop_violation
    global pass_count
    global detected_objects
    global production_count_hours
    global cycle_time_analysis
    global achieved_today

    total_count = 0
    sop_violation = 0
    pass_count = 0
    detected_objects = []
    achieved_today = "0:00:00";
    cycle_time_analysis = {
        # 0:0,
        # 1:0,
        # 2:0,
        # 3:0,
        # 4:0,
        # 5:0,
        # 6:0,
        # 7:0,
        8:'0:00:00',
        9:'0:00:00',
        10:'0:00:00',
        11:'0:00:00',
        12:'0:00:00',
        13:'0:00:00',
        14:'0:00:00',
        15:'0:00:00',
        16:'0:00:00',
        17:'0:00:00'
        # 18:0,
        # 19:0,
        # 20:0,
        # 21:0,
        # 22:0,
        # 23:0
    }
    production_count_hours = {
        # 0:0,
        # 1:0,
        # 2:0,
        # 3:0,
        # 4:0,
        # 5:0,
        # 6:0,
        # 7:0,
        8:0,
        9:0,
        10:0,
        11:0,
        12:0,
        13:0,
        14:0,
        15:0,
        16:0,
        17:0
        # 18:0,
        # 19:0,
        # 20:0,
        # 21:0,
        # 22:0,
        # 23:0
    }
    cycle_ui = {
    # 0:0,
    # 1:0,
    # 2:0,
    # 3:0,
    # 4:0,
    # 5:0,
    # 6:0,
    # 7:0,
    8:0,
    9:0,
    10:0,
    11:0,
    12:0,
    13:0,
    14:0,
    15:0,
    16:0,
    17:0
    # 18:0,
    # 19:0,
    # 20:0,
    # 21:0,
    # 22:0,
    # 23:0
    }
    return {'total_count':total_count,'sop_violation':sop_violation,'pass_count':pass_count,'production_count_hours':production_count_hours,"cycle_time_analysis":cycle_ui,"achieved_today":achieved_today}



@app.route('/selected_part_number',methods=['GET'])
def selected_part_number():
    global detection_type
    detection_type = request.args['part_number']
    #print(detection_type)
    return detection_type
    


@app.route('/part_number',methods=['GET'])
def part_number():
    global drop_down
    return jsonify(drop_down)

@app.route('/completed',methods=['GET', 'POST'])
def completed():
    global total_count
    global sop_violation
    global pass_count
    global detected_objects
    global production_count_hours
    global cycle_time_analysis
    global achieved_today

    timestamp = time.strftime('%H')

    if request.args['load'] == 'false':

        if request.args['check_pass_fail'] == '1':
            pass_count = pass_count +1
        else:
            sop_violation = sop_violation + 1

        total_count = total_count + 1

        detected_objects = []

        timeList = cycle_time_analysis
        mysum = datetime.timedelta()
        for i,v in timeList.items():
            (h, m, s) = v.split(':')
            d = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
            mysum += d

        achieved_today = str(mysum)

        timeList = [cycle_time_analysis[int(timestamp)],request.args['stopwatch']]
        mysum = datetime.timedelta()
        for i in timeList:

            (h, m, s) = i.split(':')
            d = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
            mysum += d

        cycle_time_analysis[int(timestamp)] = str(mysum)

        production_count_hours[int(timestamp)] = production_count_hours[int(timestamp)] + 1


    t=cycle_time_analysis[int(timestamp)].split(':')

    cycle_ui[int(timestamp)] = total_minutes= int(t[0])*60+int(t[1])*1 +int(t[2])/60


    total_minutes= int(t[0])*60+int(t[1])*1 +int(t[2])/60

    


    return {'total_count':total_count,'sop_violation':sop_violation,'pass_count':pass_count,'production_count_hours':production_count_hours,"cycle_time_analysis":cycle_ui,"achieved_today":achieved_today}

@app.route('/kirshi',methods=['GET'])
def kirshi():
    return render_template('lucas-tvs/index.html',class_name_dict=class_name_dict, detected_objects=detected_objects,planned_cycle=planned_cycle,total_count=total_count,pass_count=pass_count,sop_violation=sop_violation,drop_down=drop_down,length_key=length)
    

@app.route('/section3_data_endpoint', methods=['GET'])
def get_section3_data():
    # return render_template('Updated section2.html', class_name_dict=class_name_dict, detected_objects=detected_objects)
    return render_template('lucas-tvs/detect.html', class_name_dict=class_name_dict, detected_objects=detected_objects,length_key=length)



@app.route('/section2_data_endpoint', methods=['GET'])
def get_section2_data():
    return render_template('Updated section2.html', class_name_dict=class_name_dict, detected_objects=detected_objects,length_key=length)
    # return render_template('lucas-tvs/detect.html', class_name_dict=class_name_dict, detected_objects=detected_objects)



if __name__ == "__main__":
    app.run(debug=True)#debug=True

