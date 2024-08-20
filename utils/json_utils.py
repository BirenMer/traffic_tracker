import os
from utils.conversion_utils import csv_to_list
from utils.checker_utils import compare_strings

def voilation_capture_json_creator(violation_ids,main_record_file_path):
    """
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

def final_json_processor(main_violation_tracker_json,violation_json,string_to_replace='threashold_lp_images'):
    """
    json utils
    """   
    current_dir=os.getcwd()
    
    fall_back_lp=os.path.join(current_dir,'system_fall_back_assets','fall_back_normal_lp.png')
    fall_back_lp_scanned=os.path.join(current_dir,'system_fall_back_assets','fall_back_scanned_lp.png')
    fall_back_number_plate_number='MH40BP4321'

    str_to_rep=f'{string_to_replace}'

    scanned_liscense_plate_path=''
    for i in main_violation_tracker_json:
        strings=[]
        if main_violation_tracker_json[i] and main_violation_tracker_json[i]['violation_type_id']:    
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
    main_json={}
    with open(csv_file_path,'r') as file:
        contents=file.readlines()
    splitted_file_contetns=[content.split(',') for content in contents ]
    
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