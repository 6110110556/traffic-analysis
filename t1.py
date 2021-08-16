from os import name
import torch
import cv2 as cv
import time

torch.cuda.is_available()
INPUT_FILE = 'count02.mp4'

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
            print(xmin, ymin, xmax, ymax)
            (obj_center_x, obj_center_y) = (int(xmin + (xmax//2)), int(ymin + (ymax//2)))
            cv.circle(roi,(xmin, xmax), 5, (0,0,255), 2)
            cv.rectangle(roi, (xmin,ymin), (xmax, ymax), (0, 255, 255), 2)
            # text_new_object = "New " + label_object+ ' : ' + "{:.2f}".format(conf_object) + ' %' + ' L : ' + str(object_center[2])
            text_new_object = name + "{:.2f}".format(conf)
            cv.putText(roi, text_new_object, (xmin, ymin-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            

    fps = 1./(time.time()-time_start)
    cv.putText(image, "FPS: {:.2f}".format(fps), (5,30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
    cv.imshow('frame', image)

    if(cv.waitKey(1) & 0xFF==ord('q')):
       break

cv.destroyAllWindows()
cap.release()