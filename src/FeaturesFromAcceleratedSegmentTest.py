import cv2
import numpy as np

gray_image = cv2.imread('../images/beach.png', 0)

# fast = cv2.FastFeatureDetector()
fast = cv2.FastFeatureDetector_create()

# Detect keypoints
keypoints = fast.detect(gray_image, None)
print("Number of keypoints with non max suppression:"+str(len(keypoints)))

# Draw keypoints on top of the input image
img_keypoints_with_nonmax = cv2.drawKeypoints(gray_image, keypoints, None, color=(0,255,0))
cv2.imshow('FAST keypoints - with non max suppression', img_keypoints_with_nonmax)

# Disable nonmaxSuppression
# fast.setBool('nonmaxSuppression', False)
fast.setNonmaxSuppression(0)

# Detect keypoints again
keypoints = fast.detect(gray_image, None)

print("Total Keypoints without nonmaxSuppression:"+str(len(keypoints)))

# Draw keypoints on top of the input image
img_keypoints_without_nonmax = cv2.drawKeypoints(gray_image, keypoints, None, color=(0,255,0))
cv2.imshow('FAST keypoints - without non max suppression', img_keypoints_without_nonmax)
cv2.waitKey()
