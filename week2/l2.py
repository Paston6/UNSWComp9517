# Sample solution for lab task 3 (SIFT robustness to changes in rotation)

import cv2
import math
import numpy as np

from compute_sift import SiftDetector

import sys


# Rotate an image
#
# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, x, y, angle):
    rot_matrix = cv2.getRotationMatrix2D((x, y), angle, 1.0)
    h, w = image.shape[:2]

    return cv2.warpAffine(image, rot_matrix, (w, h))


# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    height, width = image.shape[:2]
    center = height // 2, width // 2
    
    return center


if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    image = cv2.imread("road_sign.jpg")
    # image = cv2.imread("NotreDame.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rotate_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = SiftDetector()

    # Store SIFT keypoints of original image in a Numpy array
    kp = sift.detector.detect(gray, None)
    kp1, des1 = sift.detector.detectAndCompute(gray, None)
    gray = cv2.drawKeypoints(gray, kp1, None)

    # Rotate around point at center of image. 'img_center' is in (Y, X) order.
    img_center = get_img_center(gray)
    x_coord = img_center[1]
    y_coord = img_center[0]

    # Degrees with which to rotate image
    angle = 135

    # Number of times we wish to rotate the image
    rotations = int(360 / 45) - 1
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()

    for i in range(rotations):
        # Rotate image
        rotate_gray = rotate(rotate_gray, x_coord, y_coord, angle)

        # Compute SIFT features for rotated image
        kp2, des2 = sift.detector.detectAndCompute(rotate_gray, None)
        kp_gray = cv2.drawKeypoints(rotate_gray, kp2, None)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        result = cv2.drawMatchesKnn(
                gray, kp1,
                rotate_gray, kp2,
                good, None, flags=2)
        
        # new_img = cv2.resize(result, (442, 373))
        cv2.imshow("Matched points", result)
        # cv2.imshow("Matched points", new_img)
        cv2.waitKey(0)
