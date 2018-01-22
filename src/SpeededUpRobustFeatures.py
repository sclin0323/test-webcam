import cv2
import numpy as np

img = cv2.imread('../images/beach.png')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#surf = cv2.SURF()
surf = cv2.xfeatures2d.SURF_create()

# This threshold controls the number of keypoints
#surf.hessianThreshold = 15000
surf.setHessianThreshold(15000)

kp, des = surf.detectAndCompute(gray, None)

img = cv2.drawKeypoints(img, kp, None, (0,255,0), 4)

cv2.imshow('SURF features', img)
cv2.waitKey()
