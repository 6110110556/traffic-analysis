# from threading import Thread
# import cv2


# class RTSPVideoWriterObject(object):
#     def __init__(self, src=0):
#         # Create a VideoCapture object
#         self.capture = cv2.VideoCapture(src)

#         # Default resolutions of the frame are obtained (system dependent)
#         self.frame_width = int(self.capture.get(3))
#         self.frame_height = int(self.capture.get(4))

#         # Set up codec and output video settings
#         self.codec = cv2.VideoWriter_fourcc(*'mp4v')
#         self.output_video = cv2.VideoWriter('video_capture/test03.mp4', self.codec, 30, (self.frame_width, self.frame_height))

#         # Start the thread to read frames from the video stream
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()

#     def update(self):
#         # Read the next frame from the stream in a different thread
#         while True:
#             if self.capture.isOpened():
#                 (self.status, self.frame) = self.capture.read()

#     def show_frame(self):
#         # Display frames in main program
#         if self.status:
#             cv2.imshow('frame', self.frame)

#         # Press Q on keyboard to stop recording
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             self.capture.release()
#             self.output_video.release()
#             cv2.destroyAllWindows()
#             exit(1)

#     def save_frame(self):
#         # Save obtained frame into video output file
#         self.output_video.write(self.frame)

# if __name__ == '__main__':
#     rtsp_stream_link = "https://sn.rtsp.me/4xi8jN01ZKlgnNCfMmGgog/1629796775/hls/kbr6KyBz.m3u8"
#     video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link)
#     while True:
#         try:
#             video_stream_widget.show_frame()
#             video_stream_widget.save_frame()
#         except AttributeError:
#             pass

# --------------------------------------------------

import cv2 as cv
# CONFIGURE PARAMETERS
(FPS) = (15)
(PATH_SAVE) = "video_capture/fps_15_v2_2_min.mp4"
(TIME_MIN) = (2)

cap = cv.VideoCapture("https://sn.rtsp.me/4xi8jN01ZKlgnNCfMmGgog/1629796775/hls/kbr6KyBz.m3u8")
(width) = (int(cap.get(3)))
(height) = (int(cap.get(4)))
(limit_frame_count) = (FPS*TIME_MIN*60)

# video  = cv.VideoWriter('video.mp4', -1, 25, (640, 480))
video = cv.VideoWriter(PATH_SAVE, cv.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
frame_count = 0

while(frame_count < limit_frame_count):
    frame_count += 1
    ret, frame = cap.read()
    video.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
    print("FRAME: ", frame_count)
cap.release()
video.release()
cv.destroyAllWindows()



# import cv2
# # CONFIGURE
# (FPS) = (30)
# (W) = (640)
# (H) = (480)

# (PATH_SAVE) = "video_capture/test01.mp4"


# # cap = cv2.VideoCapture("https://rtsp.me/embed/rzsr6db4/")
# cap = cv2.VideoCapture("video_capture/test01.mp4")

# # video  = cv2.VideoWriter('video.avi', -1, 25, (640, 480))
# video = cv2.VideoWriter(PATH_SAVE, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
# count = 0
# while(1):
#     count += 1
#     ret, frame = cap.read()
#     video.write(frame)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
#     print("FRAME: ", count)
# cap.release()
# cv2.destroyAllWindows()