import numpy as np
import cv2

def blur_except_rectangle(image, x, y, width, height, blur_kernel_size=(8, 8),offset_on=True,offset_size=[30,30]):
    """
    Function to blur a specific rectangle 
    """
    # Create a mask
    mask = np.zeros_like(image)
    if offset_on:
        cv2.rectangle(mask, (x-offset_size[0], y-offset_size[1]), (width+offset_size[0],height+offset_size[1]), (255, 255, 255), -1)
    else:
        cv2.rectangle(mask, (x, y), (width, height), (255, 255, 255), -1)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the image
    blurred_image = cv2.blur(image, blur_kernel_size)
    result = cv2.bitwise_and(image, mask) + cv2.bitwise_and(blurred_image, mask_inv)

    return result