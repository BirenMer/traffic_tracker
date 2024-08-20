import os
import shutil

from utils.conversion_utils import list_to_string
from utils.date_time_utils import get_current_date
from utils.sorting_utils import *

def file_writer(comma_string,file_name,mode='append'):
    """
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


def detection_coordinate_write(frame_count,object_id,coordinate_list,object_class,file_name):
    """
    Used to write detection coordinates
    Order frame_count,object_id,coordinate_string,object_class
    """
    coordinate_string=list_to_string(coordinate_list)
    comma_string=f'{frame_count},{object_id},{coordinate_string},{object_class}'
    file_writer(comma_string,file_name)
    return file_name

def create_directory(directory_path_or_name,inside_parent_dir=True,parent_dir_name="infenrence_output"):
    """
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

# Now we need a function which can create folders according to the id and provide us a completed path list
def evidance_directories_creator(main_violation_json):
    """
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

def frame_copier(image_list,source_folder_path,destination_folder_path):
    """
    Used to copy multiple frame to a destions from a source using frame number 
    """
    for j in image_list:
            image_name=f'{j[0]}.jpg'
            source_path=os.path.join(source_folder_path,image_name)
            destination_path=os.path.join(destination_folder_path,image_name)
            shutil.copy2(source_path,destination_path)
        
def evidance_img_separator(evidance_img_dir_paths,source_folder_path,main_violation_json,debug=False):
    """
    Used to copy evidance images into unique folders
    """
    for i in zip(main_violation_json,evidance_img_dir_paths):
            temp_arr=main_violation_json[i[0]]
            if debug:
                print(main_violation_json[i[0]])
            frame_copier(temp_arr,source_folder_path,i[1])
    return evidance_img_dir_paths  