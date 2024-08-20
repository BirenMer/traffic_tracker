import os
from utils.conversion_utils import list_divider
from utils.sorting_utils import sort_files_by_name_byte_code
from utils.model_utils import YOLO_model_loader
from utils.plotting_utils import prediction_coordinated_hadler,prediction_element_handler

def main_process_helm1et_violation_checker(cropped_violation_image_folder_path_list,violation_frame_tracker_for_helmet_detection,main_violation_tracker_json,violation_type_json,num_threads=6,gpu_device_number=0):
    # main_violation_tracker_json[f'{object_id}']['violation_type_id'].append(violation_type_json['wo_helmet'])
    ## This should take all paths list
    ## Update the violation tracker json to contain class name as well
    ## This sholud contain a list list divider
    ### Create a fucntion to create a main list of all bike consisting json
    for i in violation_frame_tracker_for_helmet_detection:
        motorcycle_ids=[]
        motocycle_path_list=[]
        if i[1]=='motorcycle':
            motorcycle_ids.append(i[0])
        for i in motorcycle_ids:
            for j in cropped_violation_image_folder_path_list:
                idx=j.split('/')[-1]
                if i==idx:
                    motocycle_path_list.append[j]
                    
    chunks=list_divider(cropped_violation_image_folder_path_list,num_threads)
    helmet_detection_models=YOLO_model_loader('hemlet_model_path',num_threads)
    for model,chunks in zip(helmet_detection_models,chunks):
        sub_process_helmet_violation_checker()
    pass

def sub_process_helmet_violation_checker(yolo_model,image_folder_path_list,main_violation_tracker_json,object_id,violation_type_json,gpu_device:int=1):
    yolo_model_classes=yolo_model.names
    for path in image_folder_path_list:
        sorted_files=sort_files_by_name_byte_code(path)
        for image_path in sorted_files:
            results=yolo_model.predict(image_path,device=[gpu_device])
            px,conf=prediction_coordinated_hadler(results)
            x1,y1,x2,y2,object_class_id,conf=prediction_element_handler(px,conf)
            if yolo_model_classes[object_class_id]=='no_helmet':
                main_violation_tracker_json[f'{object_id}']['violation_type_id'].append(violation_type_json['no_helmet'])
    pass