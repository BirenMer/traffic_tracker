import numpy as np
import cv2
import os

from utils.sorting_utils import sort_files_by_name
from utils.file_utils import create_directory

def compute_fft(f):
    """
    Working : It computes the 2-D FFT (Fast Fourier Transform) of the input image and shifts the zero-frequency component to the center of the spectrum.
    """
    ft = np.fft.fft2(f)
    ft = np.fft.fftshift(ft)
    return ft

def gaussian_filter(kernel_size,img,sigma=1, muu=0):
    """
    Usage : It generates a 2D Gaussian filter.
    """
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
    normal = 1/(((2*np.pi)**0.5)*sigma)
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    gauss = np.pad(gauss, [(0, img.shape[0] - gauss.shape[0]), (0, img.shape[1] - gauss.shape[1])], 'constant')
    return gauss

def fft_deblur(img,kernel_size=2,kernel_sigma=8,factor='wiener',const=0.002) -> np.ndarray:
    """
    Usage : It performs image deblurring using FFT.
    """
    gauss = gaussian_filter(kernel_size,img,kernel_sigma)
    img_fft = np.fft.fft2(img)
    gauss_fft = np.fft.fft2(gauss)
    weiner_factor = 1 / (1+(const/np.abs(gauss_fft)**2))
    if factor!='wiener':
        weiner_factor = factor
    recon = img_fft/gauss_fft
    recon*=weiner_factor
    recon = np.abs(np.fft.ifft2(recon))
    return recon

def de_blur_using_fft(im) -> np.ndarray:
    """
    Usage : Converts intput image to gray scale and call the deblur funtion
    """
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    recon = fft_deblur(im,kernel_size=2,kernel_sigma=10,factor='wiener',const=0.015)
    return recon

def smooth_img(im) -> np.ndarray:
    """
    Usage : Converts intput image to gray scale and call the deblur funtion
    """
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # recon = fft_deblur(im,kernel_size=2,kernel_sigma=10,factor='wiener',const=0.015)
    # Create the kernel
    kernel = np.ones((5, 5), np.float32) / 25
    # Apply the filter
    smoothed_image = cv2.filter2D(im, -1, kernel)
    return smoothed_image 

def deblur_image(im, kernel_size=2):
    
    # Convert the image to grayscale
    blurry_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Estimate motion blur kernel using the Richardson-Lucy algorithm
    deconvolution_mat = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    deblurred_img = cv2.filter2D(blurry_gray, -1, deconvolution_mat)

    return deblurred_img

def deblur_from_path(image_path,save_path):
    img=cv2.imread(image_path)
    corrected_image=deblur_image(img)
    cv2.imwrite(f'{save_path}',corrected_image)

def deblur_images(cropped_images_pathss,main_violation_json,dir_name=None):
    
    deblur_dir_paths=[]
    
    for path,idx in zip(cropped_images_pathss,main_violation_json):
        
        temp_list=sort_files_by_name(path)
        
        if dir_name:
             final_name=f'{dir_name}_deblurred_images'
        else: 
             final_name='deblurred_images'
        
        deblur_images_dir=os.path.join(final_name,str(idx))

        deblur_dir_path=create_directory(directory_path_or_name=deblur_images_dir)
        for i in temp_list:
            frame_number=os.path.basename(i).split('.')[0]
            temp_dir_path=os.path.join(deblur_dir_path,f'{frame_number}.jpg')
            deblur_from_path(image_path=i,save_path=temp_dir_path)
        deblur_dir_paths.append(deblur_dir_path)
    return deblur_dir_paths
