from os import name
import torch
import cv2 as cv
import time
import numpy as np
from motpy import Detection, MultiObjectTracker


torch.cuda.is_available()
INPUT_FILE = 'count02.mp4'
INPUT_FILE = "video_capture/fps_15_v1_1_min.mp4"

model = torch.hub.load('C:/project/Y5', 'custom', path='config_model/yolov5s.pt', source='local') 

def list_obj_detection(name):
    if(name == 'car' or name == 'bus' or name == 'truck' or name == 'motorcycle'):
        return True
    else:
        return False

cap = cv.VideoCapture(INPUT_FILE)
ret, image = cap.read()

roi_xmin, roi_ymin, x, y = cv.selectROI("AREADETECTION", image, False)
roi_xmax = roi_xmin + x
roi_ymax = roi_ymin + y
print("ROI (xmin, ymin, xmax, ymax)")
print(roi_xmin, roi_ymin, roi_xmax, roi_ymax)

while True:
    ret, image = cap.read()
    if image is None:
        print('Complete')
        break

    time_start = time.time()
    roi = image[roi_ymin: roi_ymax, roi_xmin: roi_xmax]
    cv.rectangle(image, (roi_xmin,roi_ymin), (roi_xmax, roi_ymax), (0, 255, 255), 2)
    cv.putText(image, "AREA DETECTION", (roi_xmin, roi_ymin-5) , cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    
    results = model(roi, size=640)  # includes NMS
    # Results
    results.print()
    # print(results.pandas().xyxy[0])  # img1 predictions (pandas)
    # 

    for i, obj in results.pandas().xyxy[0].iterrows():
    #     # conf = obj[4]
    #     print(obj['confidence'])
        if(obj['confidence'] > 0.3 and list_obj_detection(obj['name']) ):
            (xmin, ymin) = (int(obj['xmin']), int(obj['ymin']))
            (xmax, ymax) = (int(obj['xmax']), int(obj['ymax']))
            (conf, name) = (obj['confidence'], obj['name'])
            (x_length) = (abs(xmax - xmin))
            (y_length) = (abs(ymax - ymin))
            # print(xmin, ymin, xmax, ymax)
            print(x_length, y_length)
            (obj_center_x, obj_center_y) = (int(xmin + (x_length//2)), int(ymin + (y_length//2)))

            # create a simple bounding box with format of [xmin, ymin, xmax, ymax]
            object_box = np.array([xmin, ymin, xmax, ymax])
            # create a multi object tracker with a specified step time of 100ms
            tracker = MultiObjectTracker(dt=0.1)

            for step in range(5):
                # let's simulate object movement by 1 unit (e.g. pixel)
                object_box += 1

                # update the state of the multi-object-tracker tracker
                # with the list of bounding boxes
                tracker.step(detections=[Detection(box=object_box)])

                # retrieve the active tracks from the tracker (you can customize
                # the hyperparameters of tracks filtering by passing extra arguments)
                tracks = tracker.active_tracks()

                print('MOT tracker tracks %d objects' % len(tracks))
                # print('first track box: %s' % str(tracks[0].box))
                t_xmin = tracks[0].box[0]
                t_ymin = tracks[0].box[1]
                t_xmax = tracks[0].box[2]
                t_ymax = tracks[0].box[3]

                print(tracks)

                # print(t_xmax, t_ymax, t_xmin, t_ymin)


                cv.circle(roi,(obj_center_x, obj_center_y), 5, (0,0,255), 2)
                cv.rectangle(roi, (xmin,ymin), (xmax, ymax), (0, 255, 255), 2)
                # text_new_object = "New " + label_object+ ' : ' + "{:.2f}".format(conf_object) + ' %' + ' L : ' + str(object_center[2])
                text_new_object = name + "{:.2f}".format(conf)
                cv.putText(roi, text_new_object, (xmin, ymin-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            # cv.circle(roi,(obj_center_x, obj_center_y), 5, (0,0,255), 2)
            # cv.rectangle(roi, (xmin,ymin), (xmax, ymax), (0, 255, 255), 2)
            # # text_new_object = "New " + label_object+ ' : ' + "{:.2f}".format(conf_object) + ' %' + ' L : ' + str(object_center[2])
            # text_new_object = name + "{:.2f}".format(conf)
            # cv.putText(roi, text_new_object, (xmin, ymin-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            

    fps = 1./(time.time()-time_start)
    cv.putText(image, "FPS: {:.2f}".format(fps), (5,30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
    cv.imshow('frame', image)

    if(cv.waitKey(1) & 0xFF==ord('q')):
       break

cv.destroyAllWindows()
cap.release()