import cv2
import numpy as np
import argparse
import math
from matplotlib import pyplot as plt

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
    _, frame = cv2.threshold(frame, 50, 230, cv2.THRESH_BINARY_INV)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    return frame


def createSharpFrame(frame):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    imgLaplacian = cv2.filter2D(frame, cv2.CV_32F, kernel)
    sharp = np.float32(frame)
    imgResult = sharp - imgLaplacian
    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    #cv.imshow('Laplace Filtered Image', imgLaplacian)
    #cv2.imshow('New Sharped Image', imgResult)

    return imgResult

def waterShedFrame(frame, opening, imgResult):
    kernel = np.ones((5,5),np.uint8)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
 
    # Finding sure foreground area
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    #cv2.imshow('Distance Transform Image', dist)

    _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
    dist = cv2.dilate(dist, kernel)
    #cv2.imshow('Peaks', dist)

    dist_8u = dist.astype('uint8')
    # Find total markers
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i+1), -1)

    # Draw the background marker
    cv2.circle(markers, (5,5), 3, (255,255,255), -1)
    markers_8u = (markers * 10).astype('uint8')
    #cv2.imshow('Markers', markers_8u)

    cv2.watershed(frame, markers)
    mark = markers.astype('uint8')
    mark = cv2.bitwise_not(mark)
    cv2.imshow('Markers_v2', mark)


    



# Main --------------------------------------------------------------------
cap = getVideo("./media/TestVideo.mp4")
delay = createDelay(cap)


while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Createing sharp image
    sharp_frame = createSharpFrame(frame)

    # Processing frame
    threshed_frame = processFrame(frame, 0.4)
    watershed_frame = waterShedFrame(frame.copy(), threshed_frame, sharp_frame)

    contours, _ = cv2.findContours(threshed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours: 
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x + w, y+ h), (0, 255, 0), 3)



    # Display the frame
    cv2.imshow('Thresh Frame', threshed_frame)
    #cv2.imshow('Frame', frame)

    # Key event listener to end video replay
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()