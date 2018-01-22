import cv2
import numpy as np

gray_image = cv2.imread('../images/beach.png', 0)

# Initiate FAST detector
# fast = cv2.FastFeatureDetector()
fast = cv2.FastFeatureDetector_create()

# Initiate BRIEF extractor
#brief = cv2.DescriptorExtractor_create("BRIEF")

brief = cv2.xfeatures2d.SIFT_create()
#cv2.bgsegm.createBackgroundSubtractorMOG()
#cv2.bgsegm.createBackgroundSubtractorGMG()


# find the keypoints with STAR
keypoints = fast.detect(gray_image, None)

# compute the descriptors with BRIEF
keypoints, descriptors = brief.compute(gray_image, keypoints)

gray_keypoints = cv2.drawKeypoints(gray_image, keypoints, None, color=(0,255,0))
cv2.imshow('BRIEF keypoints', gray_keypoints)
cv2.waitKey()
