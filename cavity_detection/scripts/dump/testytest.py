import numpy as np
import cv2

H = 600
W = 400

# Show the inliers on a blank image
blank = np.zeros((H, W), dtype=np.uint8)
blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)

p1 = (10, 100)
p2 = (200, 200)

cv2.line(blank, p1, p2, (255,0,0), 1)
cv2.circle(blank, p1, 5, (0,255,0), -1)
cv2.imshow('blank', blank)
cv2.waitKey(0)