import cv2
import numpy as np

def crop_image_with_image_ndarray(image:np.ndarray,x1:float,y1:float,w:float,h:float,save_dir_path:str='')->np.ndarray:
            cropped_image = image[y1:y1+h, x1:x1+w]
            cv2.imwrite(f'{save_dir_path}',cropped_image)
            return cropped_image

def crop_image_with_image_path(image_path:str,x1:float,y1:float,w:float,h:float,save_dir_path:str='')->np.ndarray:
            image=cv2.imread(image_path)
            cropped_image = image[y1:y1+h, x1:x1+w]
            cv2.imwrite(f'{save_dir_path}',cropped_image)
            return cropped_image

def add_threshold_to_image_ndarray(img:np.ndarray,save_image_path:str=None)->np.ndarray:
    """
    image utils
    """
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray, 0, 220, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite(f'{save_image_path}',threshold_img)
    return threshold_img