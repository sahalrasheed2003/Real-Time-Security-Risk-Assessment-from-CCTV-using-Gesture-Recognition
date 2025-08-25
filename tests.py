
# import cv2
# import torch
# from ultralytics import YOLO
#
# # Load YOLO model
# yolo_model = YOLO("yolov8n-pose.pt")
#
# # Load video
# video_path = r"D:\project\Risk_assessment\myapp\static\dataset\0\nofi002.mp4"  # Replace with your video file
# cap = cv2.VideoCapture(video_path)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Perform YOLO fight detection
#     results = yolo_model(frame)
#
#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box[:4])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, "FIGHT DETECTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#     # Display the output
#     cv2.imshow("Fight Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
import cv2
import torch
from torchinfo import summary
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val)
