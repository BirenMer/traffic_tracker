
import cv2
import gc
import numpy as np

from cv2 import dnn_superres

def enhance_image_using_dnn_superres(image_path:str, re_upscale=False, model_path=None,save_dir_path=None) -> np.ndarray:
    image=cv2.imread(image_path)
    image_shape=image.shape
    if image_shape[0] >=85 and image_shape[1] >= 65:
        sr = dnn_superres.DnnSuperResImpl_create()
        if model_path:
            sr.readModel(model_path)
            sr.setModel("edsr", 4)
            upscaled_img = sr.upsample(image)
            if re_upscale:
                re_upscaled_img = sr.upsample(upscaled_img)
                final_image=re_upscaled_img
            else:
                final_image=upscaled_img
            cv2.imwrite(f'{save_dir_path}', final_image)
            # Release resources
            sr = None
            gc.collect()  # Explicitly call garbage collector
            return final_image