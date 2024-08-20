import numpy as np
import threading
import multiprocessing
import os
from utils.file_utils import create_directory
from utils.sorting_utils import sort_files_by_name_byte_code
from utils.dnn_superres_utils import enhance_image_using_dnn_superres


        
def sub_process_image_enhance_using_dnn_supress_cpu(image_path_list,queue,dir_name):
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
            enhance_image_using_dnn_superres(image_path=i,save_dir_path=temp_dir_path,model_path='models/EDSR_x4.pb')
        enhance_dir_paths.append(enhance_dir_path)
        queue.put(enhance_dir_path) 

# Default 8 for max effiency for better results
def main_process_image_enhance_using_dnn_supress_cpu(main_list,num_threads=8,dir_name=None):
    # All paths
    main_path_list=[]
    processes=[]
    queue = multiprocessing.Queue()  # Create a multiprocessing Queue
    # Dividing the list into 4 sub list
    chunks = np.array_split(main_list, int(num_threads))
    for chunk in chunks:
            p = multiprocessing.Process(target=sub_process_image_enhance_using_dnn_supress_cpu, args=[chunk, queue,dir_name])
            p.start()
            processes.append(p)
    for process in processes:
        process.join()
    while not queue.empty():
        main_path_list.append(queue.get())
    return main_path_list