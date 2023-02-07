# import torch
# import numpy as np
# import cv2
# class VideoCamera(object):
#     model = torch.hub.load('ultralytics/yolov5', 'custom',
#                         path='../yolo_v5/yolov5/runs/train/exp9/weights/best.pt', force_reload=True)
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             continue
#         frame = cv2.resize(frame, (800, 480))
#         results = model(frame)
#         # print(np.array(results.render()).shape)
#         cv2.imshow('YOLO COCO 03 mask detection', np.squeeze(results.render()))
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()


import cv2

class VideoCamera(object):
    def __init__(self):
        #由opencv來獲取預設為0 裝置影像
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()