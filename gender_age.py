"""
PyPower Projects
Detect Gender and Age using Artificial Intelligence
"""

# Usage:
# Step 1: Go to command prompt and set working directory where gender_age.py is stored
# Step 2: For image: python gender_age.py -i 1.jpg
# Step 3: For webcam: python gender_age.py

import cv2 as cv
import math
import time
import argparse
import os


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                         (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(
    description='Run age and gender recognition using OpenCV.')
parser.add_argument("-i", help='Path to input image or video file. Skip for webcam.')

args = parser.parse_args()

# Model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNetFromCaffe(ageProto, ageModel)
genderNet = cv.dnn.readNetFromCaffe(genderProto, genderModel)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Create detected folder if not exists
os.makedirs('./detected', exist_ok=True)

# Open image/video/webcam
cap = cv.VideoCapture(args.i if args.i else 0)
padding = 20

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)

    if not bboxes:
        print("No face detected, checking next frame...")
        continue

    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(
            bbox[3] + padding, frame.shape[0] - 1),
            max(0, bbox[0] - padding):min(
            bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        print(f"Gender: {gender}, Confidence: {genderPreds[0].max():.3f}")
        print(f"Age: {age}, Confidence: {agePreds[0].max():.3f}")

        label = f"{gender}, {age}"
        cv.putText(frameFace, label, (bbox[0] - 5, bbox[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

        # Save detected image
        if args.i:
            filename = os.path.basename(args.i)
        else:
            filename = f"webcam_{int(time.time())}.jpg"

        cv.imwrite(os.path.join('./detected', filename), frameFace)

    cv.imshow("Age Gender Demo", frameFace)
    print(f"Time: {time.time() - t:.3f} sec")