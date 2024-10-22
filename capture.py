import cv2
import numpy as np
import argparse
import math
import imutils
from matplotlib import pyplot as plt
from tracking import *

# Functions ---------------------------------------------------------------
def adjust_gamma(image, gamma = 1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def getVideo(video_path):
    capture = cv2.VideoCapture(video_path) # Capture Video
    
    if not capture.isOpened():
        print("Error: Could not open video.")
        exit()

    return capture

def createDelay(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    slow_factor = 2  # Change this value to control the slow speed
    delay = int(1000 / (fps / slow_factor))
    return delay


def processFrame(frame, gamma = 1.0):
    kernel = np.ones((5,5),np.uint8)

    frame = adjust_gamma(frame, gamma)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    _, frame = cv2.threshold(frame, 80, 190, cv2.THRESH_BINARY_INV)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    edged = cv2.Canny(frame, 111, 230)
    edged = cv2.dilate(edged,kernel,iterations=1)


    return edged


def secondWaterShed(original_frame, threshed_frame):
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(threshed_frame,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)

    #cv2.imshow("sure fore", sure_fg)
    #cv2.imshow("sure fback", sure_bg)


    unknown = cv2.subtract(sure_bg,sure_fg)

    #cv2.imshow("unknown", unknown)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(original_frame, markers)
    original_frame[markers == -1] = [255,0,0]

    markers_normalized = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the normalized markers to 8-bit so that they can be displayed
    markers_8bit = np.uint8(markers_normalized)

    # Apply a colormap for better visualization
    # You can use any colormap, here we use 'cv2.COLORMAP_JET' for example
    colored_markers = cv2.applyColorMap(markers_8bit, cv2.COLORMAP_JET)

    # Display the markers with the colormap
    # cv2.imshow("Markers with Colormap", colored_markers)

    #cv2.imshow("Water2", original_frame)
    return original_frame


def canny(frame):
    kernel = np.ones((3,3),np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(frame, 30, 150)
    edged = cv2.dilate(edged,kernel,iterations=1)

    return edged


def drawBoxes(original_frame, threshed_frame):
    contours, _ = cv2.findContours(threshed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours: 
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(original_frame, (x,y), (x + w, y+ h), (0, 255, 0), 3)
    
    return original_frame


# Main --------------------------------------------------------------------
cap = getVideo("./media/TestVideo.mp4")
delay = createDelay(cap)


while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=1200)

    # Createing sharp image

    # Processing frame
    threshed_frame = processFrame(frame, 0.5)
    second_watershed_frame = secondWaterShed(frame.copy(), threshed_frame)
    canny_frame = canny(frame.copy())

    # boxes = returnBoxes(frame)

    frame2 = frame.copy()
    frame3 = frame.copy()
    frame2 = drawBoxes(frame2, canny_frame)
    frame3 = drawBoxes(frame3, threshed_frame)


    # Display the frame
    cv2.imshow('Thresh Frame', threshed_frame)
    #cv2.imshow('WaterShed Frame', watershed_frame)
    cv2.imshow("Canny Frame", canny_frame)

    # boxes = returnBoxes(frame)
    # for box in boxes:
    #     (x, y, w, h) = [int(v) for v in box]
    #     cv2.rectangle(frame3, (x, y), (x + w, y + h), (0, 255, 0), 2)


    
    cv2.imshow('Boxes2', frame2)
    cv2.imshow('Boxes3', frame3)

    # Key event listener to end video replay
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()