from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np 
import cv2 
import imutils

# statico
#image = cv2.imread("assets/caminando.jpg")

cap = cv2.VideoCapture('assets/video_5.mp4');

while True:
    ret,frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 1.0
    w = int(image.shape[1] / scale)
    image = imutils.resize(image, width=w)
    #image = imutils.resize(image, width=min(400, image.shape[1]))
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    orig = image.copy()
    
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("Output", orig)
    cv2.imshow("OutputNMS", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''
  #resize image -scale
    scale = 1.0
    w = int(image.shape[1] / scale)
    image = imutils.resize(image, width=w)
    #image = imutils.resize(image, width=min(400, image.shape[1]))
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    orig = image.copy()
    
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("Output", orig)
    cv2.imshow("OutputNMS", image)
'''