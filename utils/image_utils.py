import cv2
import numpy as np
import threading
import multiprocessing

from PIL import Image

from utils.blurring_utils import blur_except_rectangle
from utils.file_utils import create_directory
from utils.sorting_utils import *
from utils.conversion_utils import coordinates_creator
from utils.general_image_utils import crop_image_with_image_path



def crop_image_using_paths(dir_paths,main_violation_json,parent_folder_name='cropped_evidance_images'):
    cropped_image_paths=[]
    for i in zip(dir_paths,main_violation_json):
        temp_list=sort_files_by_name(i[0])
        dir_path=create_directory(f'{parent_folder_name}/{i[1]}')
        temp_arrx=main_violation_json[i[1]]
        # Now we have all the images we need
        for j in zip(temp_list,temp_arrx):
            frame_number=os.path.basename(j[0]).split('.')[0]
            if frame_number==j[1][0]:
                x1,y1,x2,y2=j[1][2],j[1][3],j[1][4],j[1][5]
                x1,y1,w,h=coordinates_creator(x1,y1,x2,y2)
                final_path=os.path.join(dir_path,f'{frame_number}.jpg')
                # Function to crop image
                crop_image_with_image_path(j[0],x1=x1,y1=y1,w=w,h=h,save_dir_path=final_path)
        cropped_image_paths.append(dir_path)
    return cropped_image_paths

def frame_writer(count,frame,folder_path,target_width=720,target_height=720,resize=False):
    """
    Writes frames based on provided folder_path and frame_count
    """
    filename = f'{count}.jpg' 
    frame_type=type(frame)

    if frame_type==None :
     return "NO Image Found"
    else:
            image_write_path = os.path.join(folder_path,filename)
            main_frame=[]
            
            if resize:
                main_frame = cv2.resize(frame, (target_width, target_height))
            
            else :
                main_frame=frame
      
            if main_frame is not None:
                cv2.imwrite(image_write_path, main_frame)
                return image_write_path
            else: 
                return "No Image found"
            
def blurred_frame_writer(count,frame,folder_path,target_width=720,target_height=720,resize=False,bbox_list=[0,0,0,0]):
    """
    Writes frames based on provided folder_path and frame_count
    """
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
            main_frame=blur_except_rectangle(image=main_frame,x=x1,y=y1,width=w,height=h,offset_on=True)
                
            if main_frame is not None:
                cv2.imwrite(image_write_path, main_frame)
                return image_write_path
            else: 
                return "No Image found"

def read_image_with_PIL(image_path):
    """
    image utils
    :param image_path: String, path to the input image.
    Returns:
        image: PIL Image.
    """
    image = Image.open(image_path).convert('RGB')
    return image