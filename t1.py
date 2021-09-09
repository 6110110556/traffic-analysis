from os import name
import torch
import cv2 as cv
import time
import numpy as np


torch.cuda.is_available()

############ CONFIGURE PARAMETERS #################################
# MODEL MANAGEMENT
INPUT_FILE = "video/fps_30_v2_2_min.mp4"
INPUT_MODEL = "config_model/yolov5s.pt"
CONFIDENCE_THRESHOLD = 0.5

# FLAGE EVENT
flag_config_ROI = True

# COLOR OBJECTIVES
color_car_detection = (0, 255, 0)
color_car_tracking = (255, 255, 0)

color_truck_detection = (0, 0, 255)
color_truck_tracking = (127, 127, 127)

color_bus_detection = (255, 0, 0)
color_bus_tracking = (255, 127, 127)

color_motorcycle_detection = (255, 0, 0)
color_motorcycle_tracking = (255, 127, 127)


## TRACKING PARAMETERS
# list parameter
tracking_list_obj = None  # have [(obj_center_x, obj_center_y), life for delete tracking, flag for decrease life, (number for count car, flag to count)] 
tracking_list_delete_obj = []

# configure tracking
life = 30
tracking_x_distance = 30
tracking_y_distance = 30

# configure for count car if car tracked x round
round_to_count = 10


# OUTPUTS
count_car = 0
count_accident = 0




###################################################################

############ FUNCTIONAL ############################################
# LIST TYPE OBJECT DETECTION
def list_obj_detection(name):
    if(name == 'car' or name == 'bus' or name == 'motorcycle'):
        return True
    else:
        return False

# SELECT COLOR FOR OBJECTIVE
def select_color_object(name):
    if(name == 'car'):
        return color_car_detection, color_car_tracking
    if(name == 'bus'):
        return color_bus_detection, color_bus_tracking
    # if(name == 'truck'):
    #     return color_truck_detection, color_truck_tracking
    if(name == 'motorcycle'):
        return color_motorcycle_detection, color_motorcycle_tracking
    else:
        return (0, 255, 0), (0, 255, 255)

# DRAWING BOUNDING BOXES
def draw_box_obj(color, center_x, center_y, d_xmin, d_xmax, d_ymin, d_ymax, text ):
    cv.circle(frame,(center_x, center_y), 1, (0,0,255), 2)
    cv.rectangle(frame, (d_xmin, d_ymin), (d_xmax, d_ymax), (0, 255, 0), 2)
    cv.putText(frame, text, (d_xmin, d_ymin-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# UPDATE POSITION LIFE POINT OF COUNTING
def update_tracking_obj(index, center_x, center_y):
        tracking_list_obj[index][0] = (center_x, center_y)
        tracking_list_obj[index][1] = life
        tracking_list_obj[index][2] = True
        tracking_list_obj[index][3] += 1

    

###################################################################


# RUN MODEL 
model = torch.hub.load('F:/Project/traffic-analysis', 'custom', path=INPUT_MODEL, source='local') 
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

cap = cv.VideoCapture(INPUT_FILE)
ret, image = cap.read()

# select ROI 
if flag_config_ROI is True:
    roi_xmin, roi_ymin, x, y = cv.selectROI("AREADETECTION", image, False)
    roi_xmax = roi_xmin + x
    roi_ymax = roi_ymin + y
    print("ROI (xmin, ymin, xmax, ymax)")
    print(roi_xmin, roi_ymin, roi_xmax, roi_ymax)

# Process
while True:
    ret, frame = cap.read()
    if frame is None:
        print('Complete')
        break
    full_frame = frame.copy()

    time_start = time.time()
    if flag_config_ROI is True:
        frame = full_frame[roi_ymin: roi_ymax, roi_xmin: roi_xmax]
        cv.rectangle(full_frame, (roi_xmin,roi_ymin), (roi_xmax, roi_ymax), (0, 255, 255), 2)
        cv.putText(full_frame, "AREA DETECTION", (roi_xmin, roi_ymin-5) , cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    
    results = model(frame, size=640)  # includes NMS
    
    
    # Results
    # results.print()
    # print(results.pandas().xyxy[0])  # img1 predictions (pandas)

    for idx, obj in results.pandas().xyxy[0].iterrows():

        if(obj['confidence'] > 0.3 and list_obj_detection(obj['name']) ):
            # define parameters
            (xmin, ymin) = (int(obj['xmin']), int(obj['ymin']))
            (xmax, ymax) = (int(obj['xmax']), int(obj['ymax']))
            (conf, name) = (obj['confidence'], obj['name'])
            (x_length) = (abs(xmax - xmin))
            (y_length) = (abs(ymax - ymin))
            # print(xmin, ymin, xmax, ymax)
            # print(x_length, y_length)
            (obj_center_x, obj_center_y) = (int(xmin + (x_length//2)), int(ymin + (y_length//2)))
            color_detection, color_tracking = select_color_object(name)
            # print("CDETECTION", color_detection, "CTRACKING", color_tracking)

            if tracking_list_obj is None:
                
                # get first new obj to tracking
                tracking_list_obj = []
                tracking_list_obj.append([(obj_center_x, obj_center_y), life, True, 0])

                # draw bounding new box obj
                text_new_object = "New " + name + ' : ' + "{:.2f}".format(conf) + ' %'
                draw_box_obj(color_detection, obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_new_object)

            else:
                # check id old object ?
                for i, track_obj in enumerate(tracking_list_obj):
                    # print("track objective", track_obj)
                    if(abs(obj_center_x - track_obj[0][0]) < tracking_x_distance and abs(obj_center_y - track_obj[0][1]) < tracking_y_distance):

                        # update position and life objective tracked
                        update_tracking_obj(i, obj_center_x, obj_center_y)
                        if (track_obj[3] == round_to_count):
                            count_car += 1

                        # draw bounding box for tracking object
                        text_tracking_object = "Tracking " + name + ' : ' + "{:.2f}".format(conf) + ' %'
                        draw_box_obj(color_tracking, obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_tracking_object)
                        flag_new_object = False
                        break
                    else:
                        flag_new_object = True
                        tracking_list_obj[i][2] = False
                
                if flag_new_object is True:
                    # new objective so adding list tracking
                    tracking_list_obj.append([(obj_center_x, obj_center_y), life, True, 0])
                    flag_new_object = False
                    # draw bounding new box obj
                    text_new_object = "New " + name + ' : ' + "{:.2f}".format(conf) + ' %'
                    draw_box_obj(color_detection, obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_new_object)

        # print("TRACKLIST::::::: ", tracking_list_obj)
        if(len(tracking_list_obj) > 1):
            # if objective not move so delete degress life one point
            for i, track_obj in enumerate(tracking_list_obj):
                # print("TRACK LIST", track_obj[0])
                if track_obj[2] is False:
                    if(track_obj[1] <= 0):
                        tracking_list_delete_obj.append(i)
                    else:
                        tracking_list_obj[i][1] -= 1
                        tracking_list_obj[i][2] = True 
            
            if(len(tracking_list_delete_obj) > 0):
                # print("LIST DELETE", tracking_list_delete_obj)
                for i, list_del in reversed(list(enumerate(tracking_list_delete_obj))):
                    print(i, "index delete : " ,list_del)
                    tracking_list_obj.pop(list_del)
                # reset list delete tracking obj
                tracking_list_delete_obj = []

    fps = 1./(time.time()-time_start)
    cv.putText(full_frame, "FPS: {:.2f}".format(fps), (5,30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
    cv.putText(full_frame, "COUNT: " + str(count_car), (5,60), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
    cv.imshow('Full frame', full_frame)

    if(cv.waitKey(1) & 0xFF==ord('q')):
       break

print("COUNT CAR : ",count_car)
cap.release()
cv.destroyAllWindows()
