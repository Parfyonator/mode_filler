import cv2
import numpy as np
from scipy.stats import mode


# load images and add padding of 1 pixel
color_mask = cv2.imread('assets/color_mask.png')
padded_color_mask = cv2.copyMakeBorder(color_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
mask = cv2.imread('assets/mask.png', cv2.IMREAD_GRAYSCALE)
padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

# apply threshold to get binary image
_, binary_image = cv2.threshold(padded_mask, 200, 255, cv2.THRESH_BINARY_INV)

# find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# tmp bw image to fill region masks
region_masks = np.zeros_like(mask)

# resulting image
res_image = np.zeros_like(padded_color_mask)

# fill regions with mode colors
for i, contour in enumerate(contours):
    # clipped counter
    clipped_counter = (i % 256) + 1
    # refresh region masks
    if clipped_counter == 1:
        region_masks = np.zeros_like(mask)
    # fill region enclosed by contour with specific color
    cv2.drawContours(region_masks, [contour], -1, clipped_counter, cv2.FILLED)
    
    # get points inside the contour
    region_points = np.where(region_masks == clipped_counter)
    # those points colors
    colors = padded_color_mask[region_points]
    # mode color
    mode_color = mode(colors, axis=0).mode
    # convert to ints
    mode_color = tuple(map(int, mode_color))
    # fill the contour with the mode color
    cv2.drawContours(res_image, [contour], -1, mode_color, cv2.FILLED)

# add region boundaries as black "mesh"
boundary_pixels = np.where(binary_image != 0)
res_image[boundary_pixels] = (0, 0, 0)

# remove padding
res_image = res_image[1:-1, 1:-1]
# save image
cv2.imwrite('output.jpg', res_image)
