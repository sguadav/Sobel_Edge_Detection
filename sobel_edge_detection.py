import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from basics import convert_to_grayscale, show_image, thresholding, image_histogram
from PIL import Image


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


# def gaussian_filter(filter_size, sigma=1, verbose=False):
#     # filter_1D = np.linspace(-(filter_size // 2), filter_size // 2, filter_size)
#     # for i in range(filter_size):
#     #     filter_1D[i] = dnorm(filter_1D[i], 0, sigma)
#     # filter_2D = np.outer(filter_1D.T, filter_1D.T)
#     #
#     # filter_2D *= 1.0 / filter_2D.max()
#     filter_2D = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
#     # 1 2 1
#     # 2 4 2
#     # 1 2 1
#     return filter_2D


def convolution(image, filter, average=False):
    # Source: https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
    # Explanation: https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = convert_to_grayscale(image)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("filter Shape : {}".format(filter.shape))

    image_row, image_col = image.shape
    filter_row, filter_col = filter.shape

    output = np.zeros(image.shape)

    pad_height = int((filter_row - 1) / 2)
    pad_width = int((filter_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(filter * padded_image[row:row + filter_row, col:col + filter_col])
            # if average:
            #     output[row, col] /= filter.shape[0] * filter.shape[1]

    print("Output Image size : {}".format(output.shape))

    return output


def gaussian_blur(image, verbose=False):
    # Smoothing the image is key to do object detection. Smoothing is often used to reduce noise within an image
    # or to produce a less pixelated image. If you take the derivate of a noise image, the result will be more noise.
    # Source: https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
    # filter_gau = gaussian_filter(filter_size, sigma=1, verbose=False)
    filter_gau = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])            #################
    print(filter_gau)

    blurred_image = convolution(image, filter_gau, average=True)

    plt.imshow(blurred_image, cmap='gray')
    plt.title("Gaussian Blurred Image")
    plt.show()
    return blurred_image


def image_enhancement(img_matrix):
    filter_enh = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])            #################
    enhanced_image = convolution(img_matrix, filter_enh)

    plt.imshow(enhanced_image, cmap='gray')
    plt.title("Enhanced Image")
    plt.show()

    median_image = median_filter(enhanced_image, filter_size=3)
    plt.imshow(median_image, cmap='gray')
    plt.title("Median Image")
    plt.show()

    return median_image


def median_filter(img_matrix, filter_size):
    image_row, image_col = img_matrix.shape
    filter_row, filter_col = filter_size, filter_size

    output = np.zeros(img_matrix.shape)

    pad_height = int((filter_row - 1) / 2)
    pad_width = int((filter_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img_matrix

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.median(padded_image[row:row + filter_row, col:col + filter_col])

    return output


def core_sobel(image, filter_gx, filter_gy, verbose=True):
    new_image_x = convolution(image, filter_gx)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()

    new_image_y = convolution(image, filter_gy)

    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    # |G| = sqrt(Gx^2 + Gy^2)
   # gradient_magnitude *= 255.0 / gradient_magnitude.max()

    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title("Sobel Detection Image")
    plt.show()

    return gradient_magnitude

def sobel_edge_detection(img_matrix):
    # The Sobel operator performs a 2-D spatial gradient measurement on an image and so emphasizes regions of
    # high spatial frequency that correspond to edges.
    # Calculated seperatly but can be combined as |G| = sqrt(gx^2 + gy^2)
    # Sobel: https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm

    filter_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])            #################
    filter_gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])            #################

    blurred_image = gaussian_blur(img_matrix, verbose=True)
    enhanced_image = image_enhancement(blurred_image)
    edged_image = core_sobel(enhanced_image, filter_gx, filter_gy)

    # image_histogram(edged_image)
    thresholded_image = thresholding(edged_image, threshold_value=150)            #################

    plt.imshow(thresholded_image, cmap='gray')
    plt.title("Threhsolded Image")
    plt.show()

    return


image_path = "Extra Images/Hopper-3 square.jpg"
img_cv = cv.imread(image_path)
# show_image(img_cv, "Initial Image")

gray_img = convert_to_grayscale(img_cv)
plt.imshow(gray_img, cmap='gray')
plt.title("Grayscale Image")
plt.show()
median_filter(gray_img, 3)
sobel_edge_detection(gray_img)

