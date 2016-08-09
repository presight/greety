#!/usr/bin/python
# -*- coding: utf-8 -*-

##
# Find all faces on webcam and save them to aligned, cropped images
##

import cv2
import openface
import uuid
import greeter
from greeter import save_unknown_face_img, get_faces_bounding_boxes_dlib, Face

greeter.session_id = str(uuid.uuid1())
greeter.generated_image_id = 0
face_predictor_file = './openface/models/dlib/shape_predictor_68_face_landmarks.dat'
greeter.align = openface.AlignDlib(face_predictor_file)    

if __name__ == '__main__':
    video_capture_device = 1

    vc = cv2.VideoCapture(video_capture_device)

    while True:
        _, img = vc.read()
        boxes = get_faces_bounding_boxes_dlib(img)
        faces = [Face(x, None) for x in boxes]

        for face in faces:
            save_unknown_face_img(img, face)
            greeter.generated_image_id += 1
