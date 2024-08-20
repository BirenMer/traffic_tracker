#### Phase one UTILS
import cv2
import multiprocessing
import threading
import easyocr
import numpy as np

import pytesseract

from utils.ocr_utils import run_ocr
from utils.model_utils import *
from utils.sorting_utils import *
from utils.file_utils import create_directory,file_writer
from utils.plotting_utils import prediction_coordinated_hadler,finding_coordinates
from utils.general_image_utils import crop_image_with_image_ndarray
from utils.conversion_utils import list_divider,coordinates_creator
from utils.checker_utils import compare_strings,valid_license_plate
from utils.date_time_utils import get_current_date

def sub_process_detecting_and_cropping_liscense_plate(chunk,model,thread_id,queue_,debug=False,model_conf:float=0.45,gpu_device_number=1):    
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
                            cropped_img_number_plate=crop_image_with_image_ndarray(image,x1,y1,w,h,crooped_image_file_path)
                queue_.put(save_dir_path)

def main_process_detecting_and_cropping_liscense_plate(image_folder_path,model_path,num_threads,model_conf=0.45):
    """
    Pre Processign part one: 
    Respoinsible for maintaining number plate detections and cropping detected image
    """
    path_list_queue = multiprocessing.Queue()
    all_threads=[]
    main_path_list=[]
    
    LPD_models=YOLO_model_loader(model_path,num_threads)
    list_chunks=list_divider(image_folder_path,num_threads)
    
    for index,chunk in enumerate(list_chunks):
        t=threading.Thread(target=sub_process_detecting_and_cropping_liscense_plate,args=(chunk,LPD_models[index],index,path_list_queue,model_conf),daemon=True)
        t.start()
        all_threads.append(t)
    
    for thread in all_threads:
        thread.join()

    while not path_list_queue.empty():
        main_path_list.append(path_list_queue.get())

    for model in LPD_models:
        remove_model_from_gpu(model)
    return main_path_list

def sub_process_ocr_scanner_for_liscense_plate(folder_paths,csv_file_name,tr_model,tr_processor,tr_device):
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

def main_process_ocr_scanner_for_liscense_plate(folder_paths,threads=3,gpu_device_number=1):
        """anrp utils"""
        all_process_list=[]
        chunks=list_divider(folder_paths,number_of_part=threads)    
        tr_ocr_models=tr_ocr_model_loader(numthreads=threads,device_number=gpu_device_number)
        # main_queue=multiprocessing.Queue()
        current_date=get_current_date()
        csv_file_name=f"{current_date}_ocr_results.csv"
        csv_file_path=os.path.join(os.getcwd(),csv_file_name)
        for chunk,model_json in zip(chunks,tr_ocr_models):
            print(chunk)
            t=threading.Thread(target=sub_process_ocr_scanner_for_liscense_plate,args=[chunk,csv_file_name,model_json['model'],model_json['processor'],model_json['device']])
            t.start()
            all_process_list.append(t)
        for process in all_process_list:
            process.join()
        tr_ocr_model_deloader(tr_ocr_models)
        return csv_file_path