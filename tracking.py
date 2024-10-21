# import argparse
# import imutils
# import time
# import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", type=str,
# 	help="path to input video file")
# ap.add_argument("-t", "--tracker", type=str, default="kcf",
# 	help="OpenCV object tracker type")
# args = vars(ap.parse_args())

# # Creating Trackers global
# (major, minor) = cv2.__version__.split(".")[:2]

# if int(major) == 3 and int(minor) < 3:
# 	tracker = cv2.Tracker_create("csrt".upper())

# else:
#     OPENCV_OBJECT_TRACKERS = {
#         "csrt": cv2.TrackerCSRT_create,
#         "kcf": cv2.TrackerKCF_create,
#         "boosting": cv2.TrackerBoosting_create,
#         "mil": cv2.TrackerMIL_create,
#         "tld": cv2.TrackerTLD_create,
#         "medianflow": cv2.TrackerMedianFlow_create,
#         "mosse": cv2.TrackerMOSSE_create
#     }
#     tracker = OPENCV_OBJECT_TRACKERS["csrt"]()

# trackers = cv2.MultiTracker_create() # We will use csrt for future

# def returnBoxes(frame):
#     (success, boxes) = trackers.update(frame)
#     return boxes