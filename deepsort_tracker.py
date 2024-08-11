#!/usr/bin/env python
# coding: utf-8

# # Face detection in video
# 
# PyTorch code for detecting faces in the video.
# Expected Output:
# Format Video to display at the right hand corner “Face detected: X , Total Faces : Y”

# ## Importing Libraries and Video

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1
from deep_sort_realtime.deepsort_tracker import DeepSort


# reading the video
cap = cv2.VideoCapture('faces01.mp4')

# get the frames per second of the video
fps = cap.get(cv2.CAP_PROP_FPS)
fps


# use GPU if availble
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# ## DL Models for face detection and face embeddings

# This code uses pretrained MTCNN for face detection. Multi-Task Cascaded Convolutional Neural Networks (MTCNN), is a neural network that detects faces and facial landmarks in images.
# 
# facenet_pytorch provides model which returns both the cropped faces and the bounding boxes and probabilities of detected faces.
# Another advantage of this library is its efficient implementation of the model, thus producing faster results
# 
# From the cropped face images we create embeddings using InceptionResnetV1. we can also use any other CNN based model to create the embeddings. these embeddings are later used to track the faces in various frames.

mtcnn = MTCNN(keep_all=True, device=device, post_process=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# ### DeepSORT
# 
# Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT)
# 
# it has been used to track the same face in the video ie. to create a unique id of every face to help us calculate the total number of faces encountered.

# higher max_age value for longer tracking
tracker = DeepSort(max_age=40, bgr=False)


# define the codec and create VideoWriter object for writing the output video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("final_video.avi", fourcc, fps, (1280, 720))

# font for displaying the numbers
font = cv2.FONT_HERSHEY_SIMPLEX

fontScale = 1
fontColor = (255, 255, 255)
thickness = 2
lineType = 2

# keep track of the unique track ids (faces) encountered
track_ids_encountered = set()

while cap.isOpened():
    # read frames from the video
    ret, frame = cap.read()
    if not ret:
        break

    # convert the frames to PIL images with RGB channel
    frame_extracted = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # detect faces in the frame
    boxes, prob = mtcnn.detect(frame_extracted)
    faces = mtcnn(frame_extracted)

    detections = []
    if boxes is not None:
        # get the face embeddings
        embeds = resnet(faces.to(device)).detach().cpu()
        embeds = [e for e in embeds]

        # format the detections in the format (x1, y1, w, h, prob, class) as required by tracker
        detections = [
            ([x1, y1, x2 - x1, y2 - y1], prob, 1)
            for (x1, y1, x2, y2), prob in zip(boxes, prob)
        ]
        # update the tracker
        tracks = tracker.update_tracks(detections, embeds=embeds)

        # loop over the tracked detections and get the track IDs
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            
            # add to our set of encountered track IDs
            track_ids_encountered.add(track_id)

    # count the number of people in the frame
    count = len(boxes) if boxes is not None else 0

    total_count = len(track_ids_encountered)

    # display the number of faces and total faces in the frame at top right corner
    cv2.putText(
        frame,
        "Faces detected: " + str(count),
        (900, 100),
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )

    cv2.putText(
        frame,
        "Total faces: " + str(total_count),
        (900, 150),
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )

    # storing the frame in a new video
    out.write(frame)


# release the video writer and destroy all windows
cap.release()
out.release()

