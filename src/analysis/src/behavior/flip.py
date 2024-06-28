import cv2
import numpy as np

image_path = "/Users/doh/HJ/Research/prof_byeon/lizard/lizard_detection/image/undistored_img051.png"
original_image = cv2.imread(image_path)

height, width = original_image.shape[:2]

angle = -10

#rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
shear_matrix = np.array([[1, np.tan(np.radians(angle)), 0],
                         [0, 1, 0]])

#flipped_image = cv2.warpAffine(original_image, rotation_matrix, (width, height))
tilted_image = cv2.warpAffine(original_image, shear_matrix, (width, height))

cv2.imshow('Original Image', original_image)
cv2.imshow('Tilted Image', tilted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output_image.jpg', tilted_image)

