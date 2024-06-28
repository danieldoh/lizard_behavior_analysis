# This is to solve the distortion of the image.
# This uses the calibrated intrinsic camera matrix and distortion matrix.
# This will undistorted every frame before analyze it.
import cv2
import os
import numpy as np
import argparse
import logger

def solve_dist(img):
    mtx = np.array([[1.74792548e+03, 0.00000000e+00, 8.73857763e+02],
                    [0.00000000e+00, 1.63909689e+03, 9.32674108e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)

    dist = np.array([-0.51228316, 0.09036714, -0.09374357, 0.0065888, 0.01022634], dtype=np.float64)
    undistored_img = cv2.undistort(img, mtx, dist)

    return undistored_img

if __name__ == "__main__":

    #image_path = "/Users/doh/HJ/Research/prof_byeon/lizard/lizard_detection/image/img051.png"
    directory_path = "/Users/doh/HJ/Research/prof_byeon/lizard/lizard_detection"

    all_file = os.listdir(directory_path)

    video_file = [file for file in all_file if file.endswith('.mp4')]

    for video in video_file:
        cap = cv2.VideoCapture(video)

        if (cap.isOpened() == False):
            print(f"{video} is not exited or not opened properly.")

        while(cap.isOpened()):
            ret, img = cap.read()
            if ret == True:

                undistored_img = solve_dist(img)

                cv2.imwrite("", undistored_img)
               # cv2.imshow("undistored image", undistored_img)
               # cv2.waitKey(0)
               # cv2.destroyAllWindows()

