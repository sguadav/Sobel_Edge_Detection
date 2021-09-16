import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


def convert_to_grayscale(img_matrix):
    # Converting an image to grayscale using the BT.709 standard
    # Better to use grayscale when analyzing an image
    gray_weights = np.array([[[0.2126, 0.7152, 0.0722]]])
    gray_img_matrix = cv.convertScaleAbs(np.sum(img_matrix * gray_weights, axis=2))
    return gray_img_matrix


def show_image(img_matrix, img_name):
    cv.imshow(img_name, img_matrix)
    cv.waitKey(5000)
    cv.destroyAllWindows()


def image_histogram(img_matrix):
    img_matrix_flatten = img_matrix.flatten(order='C')
    # print(img_matrix_flatten)
    plt.hist(img_matrix_flatten, bins=256)
    plt.show()


def invert_image(img_matrix):
    # Invert the values of each pixel, basically doing 255 - pixel
    img_inverted = cv.bitwise_not(img_matrix)
    return img_inverted


def thresholding(img_matrix, threshold_value):
    # Simple form of doing image segmentation
    threshold_matrix = img_matrix.copy()
    threshold_matrix[threshold_matrix > threshold_value] = 255
    threshold_matrix[threshold_matrix <= threshold_value] = 0
    return threshold_matrix


# image_path = "Food Images/2-minute-microwave-chocolate-chip-pecan-cookie.jpg"
# img_cv = cv.imread(image_path)
#
# gray_img = convert_to_grayscale(img_cv)
# # image_histogram(gray_img)
# show_image(gray_img, "Gray Image")

# inverted_image = invert_image(gray_img)
# image_histogram(inverted_image)
# show_image(inverted_image, "Inverted Image")

# threshold_img = thresholding(gray_img, threshold_value=160)
# image_histogram(threshold_img)
# show_image(threshold_img, "Threshold Image")



