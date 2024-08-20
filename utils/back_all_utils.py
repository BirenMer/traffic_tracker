import numpy as np
import cv2
import pandas as pd
import os
import shutil
import gc 
import torch
import threading
import multiprocessing
import pytesseract
import easyocr
import re
import csv

from PIL import Image
from collections import Counter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datetime import date
from ultralytics import YOLO
from boxmot import StrongSORT
from boxmot import DeepOCSORT
from pathlib import Path
from cv2 import dnn_superres

def tracker_init(tracker_name:str='strong_sort',reid_model_path:str='models/osnet_x0_25_msmt17.pt',cuda_device:bool=True,cuda_device_number:int=0):
    """
    tracker_utils
    Usage: Used to initialize trackers || Default=strong_sort
               Currently support two Trackers :
               1. strong_sort (default)
               2. deep_oc_sort
    Args : 
    1. tracker_name(str) -> used to select a traker || Default = strong_sort
    2. reid_model_path(str) -> used to select a reid model  || Defualt = osnet_x0_25_msmt17
    3. cuda_device(bool) -> Used to select between CPU and GPU for calculation || Default=GPU
    4. cuda_device_number(int) -> Used to select GPU device (If there exists multiple device) || Default = 0

    Returns : Returns a tracker object 


    List of available reid model  : ['resnet50', 'resnet101', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 'hacnn', 'mlfn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0', 'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25', 'lmbn_n', 'clip']
    """
    tracker=None
    if tracker_name=='deep_oc_sort':
        tracker = DeepOCSORT(
        model_weights=Path(reid_model_path), # which ReID model to use
        device=f'cuda:{cuda_device_number}' if cuda_device else 'cpu',
        fp16=False,
        )

    else:
        tracker = StrongSORT(
        model_weights=Path(reid_model_path), # which ReID model to use
        device=f'cuda:{cuda_device_number}' if cuda_device else 'cpu',
        fp16=False,
        )
    return tracker

def custom_rectangle(image, top_left, bottom_right, corner_length=6,corner_thinkness=2, thick_color=(0, 255, 0), thin_color=(0, 0, 255)):
    """
    Plotting utils
    Draws a rectangle with bright thick corners and a thin red line for all the boundaries.

    Parameters:
    image (numpy.ndarray): The image on which to draw the rectangle.
    top_left (tuple): The top-left coordinate of the rectangle (x, y).
    bottom_right (tuple): The bottom-right coordinate of the rectangle (x, y).
    corner_length (int): The length of the thick corners. Default is 20.
    thick_color (tuple): The color of the thick corners in BGR format. Default is white (255, 255, 255).
    thin_color (tuple): The color of the thin lines in BGR format. Default is red (0, 0, 255).

    Returns:
    numpy.ndarray: The image with the rectangle drawn on it.
    """

    # Draw the thin red rectangle
    cv2.rectangle(image, top_left, bottom_right, thin_color, 1)

    # Draw the thick white corners
    # Top-left corner
    cv2.line(image, top_left, (top_left[0] + corner_length, top_left[1]), thick_color, corner_thinkness)
    cv2.line(image, top_left, (top_left[0], top_left[1] + corner_length), thick_color, corner_thinkness)

    # Top-right corner
    cv2.line(image, (bottom_right[0], top_left[1]), (bottom_right[0] - corner_length, top_left[1]), thick_color, corner_thinkness)
    cv2.line(image, (bottom_right[0], top_left[1]), (bottom_right[0], top_left[1] + corner_length), thick_color, corner_thinkness)

    # Bottom-left corner
    cv2.line(image, (top_left[0], bottom_right[1]), (top_left[0] + corner_length, bottom_right[1]), thick_color, corner_thinkness)
    cv2.line(image, (top_left[0], bottom_right[1]), (top_left[0], bottom_right[1] - corner_length), thick_color, corner_thinkness)

    # Bottom-right corner
    cv2.line(image, bottom_right, (bottom_right[0] - corner_length, bottom_right[1]), thick_color, corner_thinkness)
    cv2.line(image, bottom_right, (bottom_right[0], bottom_right[1] - corner_length), thick_color, corner_thinkness)

    return image

def plot_tracks(track,im:np.ndarray,debugging:bool=False)->np.ndarray:
        """
        Plotting utils
        Usage : Used to plot single tracks 
        Args  : 
            1. track (List) : List of coordinated and bbox related info
            2. im (np.ndarray) : image array
        Retruns : image with bbox im (np.ndarray) 
        """
        x3,y3,x4,y4,idx,confx,obj_cls,ind=track
        # print( x3,y3,x4,y4,idx,confx,cls,ind)
        x3=int(x3)
        y3=int(y3)
        x4=int(x4)
        y4=int(y4)
        idx=int(idx)

        #Plotting center point
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        bbox_list=[x3,y3,x4,y4]

        im=cv2.putText(im,f'ID - {idx}',(x3-10,y3-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
        
        # Turn the plotting on while debugging
        if debugging:
            im=cv2.circle(im,(cx,cy),3,(255,0,0),3)
    
        # Define rectangle parameters
        top_left = (x3, y3)
        bottom_right = (x4, y4)
    
        # Draw the rectangle on the image
        im = custom_rectangle(im, top_left, bottom_right)

       
        return im,idx,cx,cy,bbox_list,obj_cls

def prediction_coordinated_hadler(results):
    """
    Plotting utils
    Provides coordinates for predictions done by YOLO
    """
    data = results[0].boxes.data
    data = data.detach().cpu().numpy()
    conf_=results[0].boxes.conf
    if conf_.nelement() != 0:
        conf=(results[0].boxes.conf[0].detach().cpu().numpy().astype("float"))
    else:
        conf=0
    px = pd.DataFrame(data).astype("float")
    return px,conf

def tracker_element_handler(dets,px,conf):
    """
    Plotting utils
    Creates tracking element for and updates tracker list
    """
    d=-1
    for index, row in px.iterrows():
            temp_list=[]
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            temp_list=[x1,y1,x2,y2,d,conf]
            dets.append(temp_list)
    return dets,d

#Function to calculate Speed 
def speed_calculator_kmph(distance,elapsed_time):
    """
    calculation utils
    Args: distance : int , elasped_time: float
    Usage: To estimate the speed of the object.
    Return: speed_kmh : int (Speed in Kilometer per hour)
    """
    
    speed_kmh = (distance*2  / elapsed_time) * 3.6 # conveting this into km/h 

    return int(speed_kmh)

# Function to plot lines with labels
def line_plotter(frame,line_name,x_start,x_end,y,line_color,text_color,line_thickness:int):
    """plotting utils"""
    frame=cv2.line(frame, (x_start, y), (x_end, y), line_color, line_thickness)
    frame=cv2.putText(frame, (str(line_name)), (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return frame

def frame_writer(count,frame,folder_path,target_width=720,target_height=720,resize=False,coordiante_blur=False,bbox_list=[0,0,0,0]):
    """
    Image utils
    Writes frames based on provided folder_path and frame_count
    """
    filename = f'{count}.jpg'
    
    if coordiante_blur:
        filename=f'{count}_blurred.jpg'

    frame_type=type(frame)

    x1,y1,w,h=bbox_list

    if frame_type==None :
     return "NO Image Found"
    else:
            image_write_path = os.path.join(folder_path,filename)
            main_frame=[]
            
            if resize:
        
                main_frame = cv2.resize(frame, (target_width, target_height))
            
            else :
                main_frame=frame

            if coordiante_blur:
                main_frame=blur_except_rectangle(image=main_frame,x=x1,y=y1,width=w,height=h,offset_on=True)
                
            if main_frame is not None:
                cv2.imwrite(image_write_path, main_frame)
                return image_write_path
            else: 
                return "No Image found"

def get_current_date():
  """
  date_time_utils
  Returns the current date in DD-MM-YYYY format.
  """
  today = date.today()
  return today.strftime("%d_%m_%Y")

def file_writer(comma_string,file_name,mode='append'):
    """
    file_utils
    Modes: 
        1. append : Write while also keeping the existing content (Default)
        2. write  : Overwrites the existing content.
    """
    mode_string='a'
    if file_name.endswith('.csv'):
        file_name=file_name
    else:    
        file_name=f"{file_name}.csv"
        
    if os.path.isfile(file_name):
        if mode =='append':
            mode_string='a+'
    else:
        if mode =='overwrite':
            mode_string='w+'
        
    with open(f'{file_name}',f'{mode_string}') as file:
        file.write(f'{comma_string}\n')
        file.close()
    
def detection_coordinate_write(frame_count,object_id,coordinate_list,file_name):
    """
    file_utils
    Used to write detection coordinates
    """
    coordinate_string=list_to_string(coordinate_list)
    comma_string=f'{frame_count},{object_id},{coordinate_string}'
    current_date=get_current_date()
    final_file_name=current_date+file_name
    file_writer(comma_string,final_file_name)
    return final_file_name

def list_to_string(LIST):
    """
    conversion_utils
    Used to convert list to sting
    """
    final_string=', '.join(map(str, LIST))
    return final_string

def csv_to_list(csv_file_path):
        """
        conversion_utils
        Function to convert csv files with multiples string lines into a list Works for any csv file.
        """
        main_arr=[]
    
        with open(csv_file_path,'r') as file:
            all_frame_records=file.read()
            # Converting string to list where each entry starts with a new line
            all_frame_records=all_frame_records.splitlines()
            # Converting remain str into arr
            main_arr=[i.split(',') for i in all_frame_records]
            file.close()
        return main_arr

def create_directory(directory_path_or_name,inside_parent_dir=True,parent_dir_name="infenrence_output"):
    """
    file_utils
    Usage : Function creates direcotries from the provided path.
    Args:
    1. directory_path_or_name : Used to provide dir path or name 
    2. inside_parent_dir : Specifed a common dir name with current_date + parent_dir_name (default - inference_output)  || default = True
    3. parent_dir_name : Name for parent dir || default = infenrence_output
    """

    if inside_parent_dir:
        current_dir=os.getcwd()
        current_date=get_current_date()
        parent_dir_name=f"{current_date}_{parent_dir_name}"
        final_directory_path=os.path.join(current_dir,parent_dir_name,directory_path_or_name)
    else:
        final_directory_path=directory_path_or_name
    
    # Creating folder
    os.makedirs(final_directory_path,exist_ok=True)
    
    return final_directory_path

def voilation_capture_json_creator(violation_ids,main_record_file_path):
    """
    json_utils
    Function creates a json for all violation occured and keeps the record for each frame in which the object was appeared  
    """
    # Converting csv file to list
    main_arr=csv_to_list(main_record_file_path)
    main_json={}
    for idx in violation_ids:
        main_json[idx]=[]
        for entry in main_arr:
            # print(entry)
            if str(idx) == str(entry[1]):
                main_json[idx].append(entry)
            else:
                continue
    return main_json

def blur_except_rectangle(image, x, y, width, height, blur_kernel_size=(8, 8),offset_on=True,offset_size=[30,30]):
    """
    image_utils
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

# Now we need a function which can create folders according to the id and provide us a completed path list
def evidance_directories_creator(main_violation_json):
    """
    file_utils
    Used to create directories for storing images and vidoe clips
    """
    img_dir_paths=[]
    clip_dir_paths=[]
    
    for i in main_violation_json:
        evidance_img_path=os.path.join('evidance_images',str(i))
        evidance_clip_path=os.path.join('evidance_clips',str(i))
        img_dir_path=create_directory(evidance_img_path)        
        clip_dir_path=create_directory(evidance_clip_path)        
        img_dir_paths.append(img_dir_path)
        clip_dir_paths.append(clip_dir_path)
    
    return img_dir_paths,clip_dir_paths

def frame_copier(image_list,source_folder_path,destination_folder_path):
    """
    file_utils
    Used to copy multiple frame to a destions from a source using frame number 
    """
    for j in image_list:
            image_name=f'{j[0]}.jpg'
            source_path=os.path.join(source_folder_path,image_name)
            destination_path=os.path.join(destination_folder_path,image_name)
            shutil.copy2(source_path,destination_path)
    
def evidance_img_separator(evidance_img_dir_paths,source_folder_path,main_violation_json):
    """
    file_utils
    Used to copy evidance images into unique folders
    """

    for i in zip(main_violation_json,evidance_img_dir_paths):
            temp_arr=main_violation_json[i[0]]
            # print(main_violation_json[i[0]])
            frame_copier(temp_arr,source_folder_path,i[1])
    return evidance_img_dir_paths  

def sort_files_by_creation_time(folder_path):
    """
    sorting utils
    """
    files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            files.append((file_path, os.path.getctime(file_path)))  # Store file path and creation time

    sorted_files = sorted(files, key=lambda x: x[1])  # Sort files based on creation time
    sorted_file_paths = [file_path for file_path, _ in sorted_files]  # Extract file paths
    return sorted_file_paths

def sort_files_by_name(folder_path):
    """
    sorting utils
    """
    files = os.listdir(folder_path)
    sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
    file_paths = [os.path.join(folder_path, file) for file in sorted_files]
    return file_paths

def sort_files_by_name_byte_code(folder_path):
    """
    sorting utils
    """
    files = os.listdir(folder_path)
    sorted_files = sorted(files, key=lambda x: int(x.split(b'.')[0]))
    file_paths = [os.path.join(folder_path, file.decode('utf-8')) for file in sorted_files]
    return file_paths

def frame_copier(image_list,source_folder_path,destination_folder_path):
    """
    file_utils
    Used to copy multiple frame to a destions from a source using frame number
    """
    for j in image_list:
            image_name=f'{j[0]}.jpg'
            source_path=os.path.join(source_folder_path,image_name)
            destination_path=os.path.join(destination_folder_path,image_name)
            shutil.copy2(source_path,destination_path)
    

def frame_picker(count_id,evidance_folder_path,slider_window_size): 
    """
    file_utils
    Used to pick all the frames according the slider_window_size provided.
    """
    all_saved_frames=sort_files_by_creation_time(evidance_folder_path)
    previous_frames=[]
    post_frames=[]
    total_saved_frames=len(all_saved_frames)    
    previous_frame_counter=count_id-slider_window_size
    post_frames_counter=count_id+slider_window_size
    if count_id>slider_window_size and total_saved_frames>slider_window_size:
        for j in range(previous_frame_counter,count_id+1):
                previous_frames.append(all_saved_frames[j])
    else:
        previous_frames=all_saved_frames[:count_id]
    if total_saved_frames>post_frames_counter:
        for i in range(count_id+1,post_frames_counter):
            post_frames.append(all_saved_frames[i])    
    else:
        post_frames=all_saved_frames[count_id:-1]
    return previous_frames,post_frames

def evidance_video_creator(input_file_list,output_file_name='evidance_x',output_folder_path='evidance_clips'):
    
    """
    video utils
    """
    
    video_file_path=os.path.join(output_folder_path,f'{output_file_name}.avi')
    # object_id=video_file_path.split('/')[-1]
    # main_evidance_tracker_json[f'{object_id}']['evidence_video_url']=f'{video_file_path}'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{video_file_path}', fourcc, 20.0, (1020, 500))
    for i in input_file_list:
        frame = cv2.imread(i)
        out.write(frame)
    out.release()
    return video_file_path

def video_clip_creator(voilation_id_tracker,violation_frame_tracker,evidance_clip_dir_paths,all_frames_record_path):
    """
    video_utils

    Used to create and store video clips from the give frame paths to the given path
    """
    for i in zip(voilation_id_tracker,violation_frame_tracker,evidance_clip_dir_paths):    
        previous_frames,post_frames=frame_picker(violation_frame_tracker[i[0]],all_frames_record_path,20)
        input_frame_list=previous_frames+post_frames
        evidance_video_creator(input_frame_list,f'evidance_{i[0]}',output_folder_path=i[2])

    return evidance_clip_dir_paths

def coordinates_creator(x1,y1,x2,y2)->list:
    """
    plotting utils
    Used to create convert x1,y1,x2,y2 to x1,y1,w,h format
    """
    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    h = round(y2-y1)
    w = round(x2-x1)
    x1 = round(x1)
    y1 = round(y1)
    return [x1,y1,w,h]

def image_cropper_using_arr(image:np.ndarray,x1:float,y1:float,w:float,h:float,save_dir_path:str='')->np.ndarray:
            cropped_image = image[y1:y1+h, x1:x1+w]
            cv2.imwrite(f'{save_dir_path}',cropped_image)
            return cropped_image

def image_cropper_using_path(image_path:str,x1:float,y1:float,w:float,h:float,save_dir_path:str='')->np.ndarray:
            image=cv2.imread(image_path)
            cropped_image = image[y1:y1+h, x1:x1+w]
            cv2.imwrite(f'{save_dir_path}',cropped_image)
            return cropped_image

def evidance_cropper(evidance_img_dir_paths,main_violation_json):
    cropped_image_paths=[]
    for i in zip(evidance_img_dir_paths,main_violation_json):
        temp_list=sort_files_by_name(i[0])
        dir_path=create_directory(f'cropped_evidance_images/{i[1]}')
        temp_arrx=main_violation_json[i[1]]
        # Now we have all the images we need
        for j in zip(temp_list,temp_arrx):
            frame_number=os.path.basename(j[0]).split('.')[0]
            if frame_number==j[1][0]:
                x1,y1,x2,y2=j[1][2],j[1][3],j[1][4],j[1][5]
                x1,y1,w,h=coordinates_creator(x1,y1,x2,y2)
                final_path=os.path.join(dir_path,f'{frame_number}.jpg')
                # Function to crop image
                image_cropper_using_path(j[0],x1=x1,y1=y1,w=w,h=h,save_dir_path=final_path)
        cropped_image_paths.append(dir_path)
    return cropped_image_paths

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

def json_frame_order_checker(main_json):
    """
    checker utils
    Used to check the order of frames
    """
    maintainer=True
    for keys_ in main_json:
        order_maintained=frame_order_checker(main_json[keys_])
        print(f"order_maintainer - {order_maintained}")
        if not order_maintained:
            maintainer=False
    return maintainer

def frame_order_checker(arr):
    """
    checker_utils

    Used to check the order of frames appended in the violation json
    """
    order_maintained=True
    for cnt,i in enumerate(arr):
        if str(cnt+1) >= str(len(arr)):
            # print("finished")
            break
        if int(arr[cnt][0]) <= int(arr[cnt+1][0]):
            continue
        else:
            order_maintained=False
    return order_maintained

### Implementing MT function

def video_clip_thread(entry,violation_frame_tracker,all_frames_record_path,window_size=20,main_violation_tracker_json=None):
        """
        Video clip utils
        """
        previous_frames,post_frames=frame_picker(violation_frame_tracker[entry[0]],all_frames_record_path,window_size)
        input_frame_list=previous_frames+post_frames
        video_clip_path=evidance_video_creator(input_frame_list,f'evidance_{entry[0]}',output_folder_path=entry[2])
        main_violation_tracker_json[f'{entry[0]}']['evidence_video_url']=video_clip_path

def video_clip_creator_mt(voilation_id_tracker,violation_frame_tracker,evidance_clip_dir_paths,all_frames_record_path,main_violation_tracker_json):
    """
    video_utils
    Used to create and store video clips from the give frame paths to the given path
    """
    thread_list_=[]
    for i in zip(voilation_id_tracker,violation_frame_tracker,evidance_clip_dir_paths):    
        t=threading.Thread(target=video_clip_thread,args=[i,violation_frame_tracker,all_frames_record_path,20,main_violation_tracker_json])
        t.start()
        thread_list_.append(t)
    for j in thread_list_:
        j.join()
    return evidance_clip_dir_paths

def crop_img_threaded(file_list,coordinate_list,dir_path):
     """
     image_utils
     """
     # Now we have all the images we need
     for j in zip(file_list,coordinate_list):
            frame_number=os.path.basename(j[0]).split('.')[0]
            if frame_number==j[1][0]:
                x1,y1,x2,y2=j[1][2],j[1][3],j[1][4],j[1][5]
                x1,y1,w,h=coordinates_creator(x1,y1,x2,y2)
                final_path=os.path.join(dir_path,f'{frame_number}.jpg')
                # Function to crop image
                image_cropper_using_path(j[0],x1=x1,y1=y1,w=w,h=h,save_dir_path=final_path)

def evidance_cropper_mt(evidance_img_dir_paths,main_violation_json):
    """
    image_utils
    """
    cropped_image_paths=[]
    __evid_c_threads=[]
    for i in zip(evidance_img_dir_paths,main_violation_json):
        dir_file_list=sort_files_by_name(i[0])
        dir_path=create_directory(f'cropped_evidance_images/{i[1]}')
        id_coordinate_list=main_violation_json[i[1]]
        t=threading.Thread(target=crop_img_threaded,args=[dir_file_list,id_coordinate_list,dir_path])
        t.start()
        __evid_c_threads.append(t)
        cropped_image_paths.append(dir_path)
    for j in __evid_c_threads:
        j.join()    
    return cropped_image_paths

### List utils
def list_divider(list_to_chucnk,number_of_part):
     """
     Conversion utils
     """
     chunks = np.array_split(list_to_chucnk, int(number_of_part))
     return chunks

##### PYT utils
def remove_model_from_gpu(model_object):
    """
    Model_utils
    """
    del model_object
    gc.collect()
    torch.cuda.empty_cache() 
    return True

##### Image Enhancement utils
def enhance_image_using_paths(image_path:str, re_upscale=False, model_path=None,save_dir_path=None) -> np.ndarray:
    """
    image enhancement utils
    """
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

def enhance_images_threaded(image_path_list,queue,dir_name):
    """
    image enhancement utils
    """
    for path in image_path_list:
        enhance_dir_paths=[]
        idx=path.split('/')[-1]
        temp_list=sort_files_by_name_byte_code(path)
        if dir_name:
             final_dir_name=f'enhanced_images_{dir_name}'
        else:
             final_dir_name='enhanced_images'
        enhance_images_dir=os.path.join(final_dir_name,str(idx))
        enhance_dir_path=create_directory(directory_path_or_name=enhance_images_dir)
        for i in temp_list:
            frame_number=os.path.basename(i).split('.')[0]
            temp_dir_path=os.path.join(enhance_dir_path,f'{frame_number}.jpg')
            enhance_image_using_paths(image_path=i,save_dir_path=temp_dir_path,model_path='models/EDSR_x4.pb')
        enhance_dir_paths.append(enhance_dir_path)
        queue.put(enhance_dir_path) 

# Default 8 for max effiency for better results
def image_enhancement_using_limit_mpx(main_list,num_threads=8,dir_name=None):
    """
    image enhancement utils
    """
    # All paths
    main_path_list=[]
    processes=[]
    queue = multiprocessing.Queue()  # Create a multiprocessing Queue
    # Dividing the list into 4 sub list
    chunks = np.array_split(main_list, int(num_threads))
    for chunk in chunks:
            p = multiprocessing.Process(target=enhance_images_threaded, args=[chunk, queue,dir_name])
            p.start()
            processes.append(p)
    for process in processes:
        process.join()
    while not queue.empty():
        main_path_list.append(queue.get())
    return main_path_list

###### Defining function for anrp
def finding_coordinates(px):
    """
    plotting_utils
    """
    temp_list=[]
    for index,row in px.iterrows():  
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            temp_list=[x1,y1,x2,y2]
    return temp_list

### Model loading utils
def LPD_model_loader(model_path,num_threads):
    """
    model utils
    """
    all_models=[]
    for i in range(0,num_threads):
        model=YOLO(model_path)
        all_models.append(model)
    return all_models

# Mouse callback function
def rectangle_coordinates_calculator(img, ix, iy, fx, fy,):
        """Plotting utils"""
        coordinates_json={}
        x_start,y_start,x_end,y_end=inner_rectangle_coordinates_calculator(ix, iy, fx, fy)
        coordinates_json['initial_x']=ix
        coordinates_json['initial_y']=iy
        coordinates_json['final_x']=fx
        coordinates_json['final_y']=fy
        coordinates_json['inner_x_start']=x_start
        coordinates_json['inner_y_start']=y_start
        coordinates_json['inner_x_end']=x_end
        coordinates_json['inner_y_end']=y_end
        return coordinates_json

# Function to draw the inner rectangle centered within the outer rectangle
def inner_rectangle_coordinates_calculator(start_x, start_y, end_x, end_y,length=82):
    """Plotting utils"""
    
    # Calculate the width of the outer rectangle
    width = abs(end_x - start_x)
    
    # Calculate the center position of the outer rectangle
    center_x = (start_x + end_x) // 2
    center_y = (start_y + end_y) // 2
    
    # Calculate the coordinates for the inner rectangle
    inner_start_x = start_x
    inner_start_y = center_y - length // 2
    inner_end_x = end_x
    inner_end_y = center_y + length // 2

    # Ensure the inner rectangle stays within the bounds of the outer rectangle
    if inner_start_y < start_y:
        inner_start_y = start_y
        inner_end_y = start_y + length
    if inner_end_y > end_y:
        inner_end_y = end_y
        inner_start_y = end_y - length

    return inner_start_x,inner_start_y,inner_end_x,inner_end_y

def rectangle_plotter(img,coordinates_json):
    """Plotting utils"""
    
    img=cv2.rectangle(img, (coordinates_json['initial_x'], coordinates_json['initial_y']), (coordinates_json['final_x'], coordinates_json['final_y']), (0, 255, 0), 2)
    img=cv2.rectangle(img, (coordinates_json['inner_x_start'],coordinates_json['inner_y_start']), (coordinates_json['inner_x_end'], coordinates_json['inner_y_end']), (255, 0, 0), 2)
    return img

### Anrp functions
def sub_process_for_preprocess_one(chunk,model,thread_id,queue_,debug=False,model_conf:float=0.45,gpu_device_number=1):    
        """
        Anrp utils
        """
        if debug:
                print(f"starting thread with id : {thread_id}")
        for i in chunk:    
                file_list=sort_files_by_name_byte_code(i)
                idx=i.split('/')[-1]
                dir_path=os.path.join('cropped_number_plates',str(idx))
                save_dir_path=create_directory(dir_path)
                for img in file_list:
                    image=cv2.imread(img)
                    frame_number=os.path.basename(img).split('.')[0]
                    crooped_image_file_path=os.path.join(save_dir_path,f'{frame_number}.jpg')
                    detections = model.predict(img,save=False,verbose=False,conf=model_conf,device=[gpu_device_number])
                    if len(detections)>0:
                        px,conf=prediction_coordinated_hadler(detections)
                        temp_list=finding_coordinates(px)
                        if len(temp_list)>0:
                            x1,y1,x2,y2=temp_list
                            x1,y1,w,h=coordinates_creator(x1,y1,x2,y2) 
                            cropped_img_number_plate=image_cropper_using_arr(image,x1,y1,w,h,crooped_image_file_path)
                queue_.put(save_dir_path)

def preprocssing_part_one(image_folder_path,model_path,num_threads,model_conf=0.45):
    """
    Anrp utils   
    Pre Processign part one: 
    Respoinsible for maintaining number plate detections and cropping detected image
    """

    path_list_queue = multiprocessing.Queue()
    all_threads=[]
    main_path_list=[]
    
    LPD_models=LPD_model_loader(model_path,num_threads)
    list_chunks=list_divider(image_folder_path,num_threads)
    
    for index,chunk in enumerate(list_chunks):
        t=threading.Thread(target=sub_process_for_preprocess_one,args=(chunk,LPD_models[index],index,path_list_queue,model_conf,0),daemon=True)
        t.start()
        all_threads.append(t)

    for thread in all_threads:
        thread.join()

    while not path_list_queue.empty():
        main_path_list.append(path_list_queue.get())

    for model in LPD_models:
        remove_model_from_gpu(model)
    return main_path_list


def object_class_name_normalizer(object_class_id,custom_class_list):
    """
    CONVERSION UTILS
    """
    if 'car' in custom_class_list[object_class_id]:
        return 'car'
    
    elif 'bus' in custom_class_list[object_class_id]:
        return 'bus'
    
    elif 'truck' in custom_class_list[object_class_id]:
        return 'truck'
    
    elif 'motorcycle' in custom_class_list[object_class_id]:
        return 'motorcycle'
    
    elif 'motorcycle' in custom_class_list[object_class_id]:
        return 'motorcycle'
    
    else:
         return custom_class_list[object_class_id]
    

##### Threshold_utils

# Writing a function to create thresholded images using multiple processing 
def image_thereshold_creator(img:np.ndarray,save_image_path:str=None)->np.ndarray:
    """
    image utils
    """
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray, 0, 220, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite(f'{save_image_path}',threshold_img)
    return threshold_img
    

def list_based_threshold_image_creator(folder_path_list,main_queue):
    """
    image_utils
    """
    for i in folder_path_list:    
                file_list=sort_files_by_name_byte_code(i)
                idx=i.split('/')[-1]
                dir_path=os.path.join('threashold_lp_images',str(idx))
                save_dir_path=create_directory(dir_path)
                for img in file_list:
                    image=cv2.imread(img)
                    frame_number=os.path.basename(img).split('.')[0]
                    thresholded_img_path=os.path.join(save_dir_path,f'{frame_number}.jpg')
                    image_thereshold_creator(image,thresholded_img_path)
                main_queue.put(save_dir_path)


def create_threshold_imgs_mp(enhanced_cropped_lp_image_path,num_processes=8):
    """
    image_utils
    """
    
    all_process_list=[]
    main_path_list=[]
    main_queue=multiprocessing.Queue()
    chunks=list_divider(enhanced_cropped_lp_image_path,num_processes)

    for chunk in chunks:
        p=multiprocessing.Process(target=list_based_threshold_image_creator,args=[chunk,main_queue])
        p.start()
        all_process_list.append(p)
        
    for process in all_process_list:
        process.join()
        
    while not main_queue.empty():
        main_path_list.append(main_queue.get())
    return main_path_list


#  OCR UTILS
#### Function for MP utils
def read_image(image_path):
    """
    image utils
    :param image_path: String, path to the input image.
    Returns:
        image: PIL Image.
    """
    image = Image.open(image_path).convert('RGB')
    return image

### Sequential_function

def ocr(image, processor, model, generation_length, device):
    """anrp utils"""
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=generation_length)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def eval_new_data(image=None, model=None, processor=None, device='cpu'):
    """anrp utils"""
    generation_length = 12
    text = ocr(image, processor, model, generation_length, device)
    return text

def easy_ocr_reader_sequential(image:np.ndarray)->str:
    """anrp utils"""
    ocr_text=[]
    # enhanced_img=enhance_image(image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
        if prob > 0:
            ocr_text.append(text)
    if ocr_text!=[]:
       return ocr_text[0]
    else:
       return '' 

### Multithreaded function
#### Function for MP utils

def valid_license_plate(plate):
    """
    checker utils
    Validates if the given string is in the format of an Indian license plate.
    
    Args:
    plate (str): The license plate string to validate.
    
    Returns:
    bool: True if the plate is valid, False otherwise.
    """
    # Regular expression to match the common Indian license plate format
    pattern = r'^[A-Z]{2}\d{2}[A-Z\d]{1,4}$'
    
    # Using fullmatch to ensure the entire string matches the pattern
    return bool(re.fullmatch(pattern, plate))

def easy_ocr_reader_td(image:np.ndarray,queue=None)->str:
    """
    anrp utils
    """
    ocr_text=[]
    # enhanced_img=enhance_image(image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
        if prob > 0:
            ocr_text.append(text)
    if ocr_text!=[]:
        queue.put(ocr_text[0])
    else:
        queue.put("")

def tesserac_ocr_reader_td(image:np.ndarray,queue=None)->str:
    """
    anrp utils
    """
    text = pytesseract.image_to_string(image,lang='eng')
    queue.put(text)
    
def trocr_processor_td(img, model, processor,queue,device='cuda'):
    """
    anrp utils
    
    Process the image using a preloaded model and processor.
    """
    ocr_text = eval_new_data(image=img, model=model, processor=processor, device=device)
    queue.put(ocr_text)

def compare_strings(strings):
    """

    checker  utils
    """
    if not strings:
        return ""
    # Find the maximum length of the strings
    max_length = max(len(s) for s in strings)
    final_string = []
    for i in range(max_length):
        # Collect all crereharacters at the current position from all strings
        chars_at_pos = [s[i] for s in strings if i < len(s)]
        
        # Find the most common character at this position
        if chars_at_pos:
            most_common_char, _ = Counter(chars_at_pos).most_common(1)[0]
            final_string.append(most_common_char)
        else:
            final_string.append(' ')
    
    return ''.join(final_string)  

def tesserac_ocr_reader_sequential(image:np.ndarray)->str:
    "anrp utils"
    text = pytesseract.image_to_string(image,lang='eng')
    return text
    
def trocr_processor_sequential(img, model, processor, device='cuda'):
    """
    anrp utils
    Process the image using a preloaded model and processor.
    """
    ocr_text = eval_new_data(image=img, model=model, processor=processor, device=device)
    return ocr_text    

def tr_ocr_model_loader(numthreads=3,cuda_on=True,model_name='microsoft/trocr-small-printed',device_number=0):
    """
    Model utils
    """
    if cuda_on:
        device = f'cuda:{device_number}' if torch.cuda.is_available() else 'cpu'
    else:
        device='cpu'
    model_json_list=[]
    for _ in range(numthreads):
        model_dict={}
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        processor = TrOCRProcessor.from_pretrained(model_name)
        # Loading params into temp json
        model_dict['device']=device
        model_dict['model_name']=model_name
        model_dict['model']=model
        model_dict['processor']=processor
        model_json_list.append(model_dict)
    return model_json_list


def tr_ocr_model_deloader(model_json_list):
    """
    Model utils
    """
    for model_dict in model_json_list:
        # Move model to CPU to free GPU memory
        model_dict['model'].cpu()
        # Remove references to model and processor
        del model_dict['model']
        del model_dict['processor']
    
    # Clear the cache and run garbage collector
    torch.cuda.empty_cache()
    gc.collect()

def remove_non_alphanumeric(input_string):
    """checker utils"""
    pattern = r'[^a-zA-Z0-9]+'
    result = re.sub(pattern, '', input_string)
    return result

def run_ocr(image:np.ndarray,tr_model,tr_device,tr_processor)->str:
    """
    anrp utils
    """
    es_ocr_result=easy_ocr_reader_sequential(image)
    tes_ocr_result=tesserac_ocr_reader_sequential(image)
    tr_ocr_result=trocr_processor_sequential(image,model=tr_model,device=tr_device,processor=tr_processor)
    return [remove_non_alphanumeric(es_ocr_result),remove_non_alphanumeric(tes_ocr_result),remove_non_alphanumeric(tr_ocr_result)]

def run_ocr_mt(image:np.ndarray,tr_model,tr_device,tr_processor)->str:
    """
    anrp utils
    """

    main_queue=multiprocessing.Queue()
    strings_=[]
    
    t1=threading.Thread(target=easy_ocr_reader_td  ,args=[image,main_queue])
    t2=threading.Thread(target=tesserac_ocr_reader_td,args=[image,main_queue])
    t3=threading.Thread(target=trocr_processor_td,args=[image,tr_model,tr_device,tr_processor,main_queue])
    
    t1.start()
    t2.start()
    t3.start()
    
    t1.join()
    t2.join()
    t3.join()

    while not main_queue.empty():
        strings_.append(main_queue.get())
    
    return strings_

def ocr_sub_process_phase_three(folder_paths,csv_file_name,tr_model,tr_processor,tr_device):
        """
        anrp utils
        """
        for path in folder_paths:
            path=str(path)
            # print(type(path))
            file_list=sort_files_by_name(path)
            idx=path.split('/')[-1]
            
            for image_path in file_list:
                
                img = cv2.imread(image_path)
                #Currently we write each and every possiblity in the csv file for each Possible ID
                strings=run_ocr(img,tr_model,tr_device,tr_processor)
                final_string=compare_strings(strings)
                
                if valid_license_plate(final_string): 
                    final_csv_string=f'{idx},{final_string},{image_path}'

                else:
                    final_csv_string=f'{idx},{""},{image_path}'

                file_writer(final_csv_string,csv_file_name)

def ocr_processor_mp_phase_three(folder_paths,num_thread=3):
        """anrp utils"""
        all_process_list=[]
        chunks=list_divider(folder_paths,number_of_part=num_thread)    
        tr_ocr_models=tr_ocr_model_loader(numthreads=num_thread)
        # main_queue=multiprocessing.Queue()
        current_date=get_current_date()
        csv_file_name=f"{current_date}_ocr_results.csv"
        csv_file_path=os.path.join(os.getcwd(),csv_file_name)
        for chunk,model_json in zip(chunks,tr_ocr_models):
            print(chunk)
            t=threading.Thread(target=ocr_sub_process_phase_three,args=[chunk,csv_file_name,model_json['model'],model_json['processor'],model_json['device']])
            t.start()
            all_process_list.append(t)
        for process in all_process_list:
            process.join()
        tr_ocr_model_deloader(tr_ocr_models)
        return csv_file_path


def violation_based_json_creator(violation_arr):
    """
    Conversation utils
    """
    violation_based_id={}
    for i in violation_arr:
        violation_based_id[i['name']]=i['id']
    return violation_based_id

# Function to convert sec to HH:MM:SS time formate
def convert_seconds_to_hhmmss(seconds):
    """
    Conversation utils
    """
    hours = int(seconds // 3600) % 24  # Ensure hours do not exceed 24
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

# Function to track time in video based on FPS and Frame count
def video_time_checker(count,cap):
        """
        date time utils"""
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000.0
        current_time_formatted = convert_seconds_to_hhmmss(current_time_sec)
        return {
                "frame":count,
                "time": current_time_formatted
               }


def final_json_processor(main_violation_tracker_json,violation_json):
    """
    json utils
    """   
    current_dir=os.getcwd()
    fall_back_lp=os.path.join(current_dir,'system_fall_back_assets','fall_back_normal_lp.png')
    fall_back_lp_scanned=os.path.join(current_dir,'system_fall_back_assets','fall_back_scanned_lp.png')
    fall_back_number_plate_number='MH40BP4321'
    str_to_rep='threashold_lp_images'
    scanned_liscense_plate_path=''
    for i in main_violation_tracker_json:
        strings=[]
        if main_violation_tracker_json[i]:    
            main_violation_tracker_json[i]['numbe_plate_ocr_image_url']=fall_back_lp_scanned
            main_violation_tracker_json[i]['numbe_plate_image_url']=fall_back_lp
            main_violation_tracker_json[i]['scanned_number_plate_number']=fall_back_number_plate_number
            if i in violation_json:
                for j in violation_json[i]:
                    if j[1]!='':
                       scanned_liscense_plate_path=j[-1]
                       strings.append(j[1])
                if len(strings)>0:    
                    result=compare_strings(strings)
                    main_violation_tracker_json[i]['numbe_plate_ocr_image_url']=scanned_liscense_plate_path
                    main_violation_tracker_json[i]['numbe_plate_image_url']=scanned_liscense_plate_path.replace(str_to_rep,'enhanced_images_cropped_lp_enhanced').replace('\n','')
                    main_violation_tracker_json[i]['scanned_number_plate_number']=result 
    return main_violation_tracker_json

def ocr_result_filter(main_violation_tracker_json,csv_file_path):
    """
    json utils
    """
    main_json={}
    with open(csv_file_path,'r') as file:
        contents=file.readlines()
        # print(contents)
    splitted_file_contetns=[content.split(',') for content in contents ]
    # print(type(splitted_file_contetns[0][0]))
    for idx in main_violation_tracker_json:
            temp_list=[]
            for count,entry in enumerate(splitted_file_contetns):
                if str(idx)==str(entry[0]):
                    temp_list.append(splitted_file_contetns[count])
            if temp_list:
                main_json[f'{idx}']=temp_list
    return main_json

def remove_empty_from_dict(d):
    """
    json utils
    """
    if type(d) is dict:
        return dict((k, remove_empty_from_dict(v)) for k, v in d.items() if v and remove_empty_from_dict(v))
    elif type(d) is list:
        return [remove_empty_from_dict(v) for v in d if v and remove_empty_from_dict(v)]
    else:
        return d
    
def json_to_csv(json_data, order=None, csv_filename='inference_result.csv',debug=False): 
    """
    conversion utils
    """ 
    if not order:
        order = ['violation_type_id', 'vehicle_type_id', 'vehicle_speed', 'travel_direction', 'location_id', 'camera_id', 'evidence_image_url', 'numbe_plate_ocr_image_url', 'evidence_video_url', 'numbe_plate_image_url', 'scanned_number_plate_number', 'violation_date', 'violation_time']
    # Open the CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        if debug:
            # Write the header row
            writer.writerow(order)
        # Write the data rows
        for key in sorted(json_data.keys(), key=int):  # Ensure the keys are sorted numerically
            row = [json_data[key].get(field, '') for field in order]
            writer.writerow(row)
