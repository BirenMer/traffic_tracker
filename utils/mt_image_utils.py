import os
import threading
import multiprocessing
import cv2

from utils.file_utils import create_directory
from utils.sorting_utils import sort_files_by_name,sort_files_by_name_byte_code
from utils.image_utils import crop_image_with_image_path
from utils.conversion_utils import list_divider,coordinates_creator
from utils.general_image_utils import add_threshold_to_image_ndarray


def sub_process_crop_images_using_paths(file_list,coordinate_list,dir_path):
     # Now we have all the images we need
     for j in zip(file_list,coordinate_list):
            frame_number=os.path.basename(j[0]).split('.')[0]
            if frame_number==j[1][0]:
                x1,y1,x2,y2=j[1][2],j[1][3],j[1][4],j[1][5]
                x1,y1,w,h=coordinates_creator(x1,y1,x2,y2)
                final_path=os.path.join(dir_path,f'{frame_number}.jpg')
                # Function to crop image
                crop_image_with_image_path(j[0],x1=x1,y1=y1,w=w,h=h,save_dir_path=final_path)

def main_process_crop_images_using_paths(evidance_img_dir_paths,main_violation_json):
    """
    image_utils
    """
    cropped_image_paths=[]
    __evid_c_threads=[]
    for i in zip(evidance_img_dir_paths,main_violation_json):
        dir_file_list=sort_files_by_name(i[0])
        dir_path=create_directory(f'cropped_evidance_images/{i[1]}')
        id_coordinate_list=main_violation_json[i[1]]
        t=threading.Thread(target=sub_process_crop_images_using_paths,args=[dir_file_list,id_coordinate_list,dir_path])
        t.start()
        __evid_c_threads.append(t)
        cropped_image_paths.append(dir_path)
    for j in __evid_c_threads:
        j.join()    
    return cropped_image_paths

def sub_process_threshold_image_creator_using_paths(folder_path_list,main_queue):
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
                    add_threshold_to_image_ndarray(image,thresholded_img_path)
                main_queue.put(save_dir_path)

def main_process_threshold_image_creator_using_paths(enhanced_cropped_lp_image_path,num_processes=8):
    """
    image_utils
    """
    
    all_process_list=[]
    main_path_list=[]
    main_queue=multiprocessing.Queue()
    chunks=list_divider(enhanced_cropped_lp_image_path,num_processes)

    for chunk in chunks:
        p=multiprocessing.Process(target=sub_process_threshold_image_creator_using_paths,args=[chunk,main_queue])
        p.start()
        all_process_list.append(p)
        
    for process in all_process_list:
        process.join()
        
    while not main_queue.empty():
        main_path_list.append(main_queue.get())
    return main_path_list
