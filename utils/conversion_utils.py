import numpy as np
import csv

def list_to_string(LIST):
    """
    Used to convert list to sting
    """
    final_string=', '.join(map(str, LIST))
    return final_string

def csv_to_list(csv_file_path):
        """
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

def list_divider(list_to_chucnk,number_of_part):
     """
     Conversion utils
     """
     chunks = np.array_split(list_to_chucnk, int(number_of_part))
     return chunks

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
    
    elif 'bicycle' in custom_class_list[object_class_id]:
        return 'bicycle'
    
    else:
         return custom_class_list[object_class_id]
    
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

def csv_to_json_arr(file_path,violation_json):
    with open(file_path,'r') as f:
        contents=f.read()
    splitted_array=[i.split(',') for i in contents.split('\n')]
        
    main_arr=[]
    for i in violation_json:
        temp_json={}
        temp_json[f'{i}']=[]
        for j in splitted_array:
            if len(j) > 0 and len(j)>6:
                # print(j)
                if str(i)==str(j[1]):
                    temp_json[f'{i}'].append(j)
        main_arr.append(temp_json)
    return main_arr

def arr_to_coordinate_json(arr):
    """
    Retrun a json with keys as frame numbers and values are coordinates for each and every frame 
    """
    main_json={}
    for i  in arr:
        coord_array=[]
        coord_array.append(i[2])
        coord_array.append(i[3])
        coord_array.append(i[4])
        coord_array.append(i[5])
        main_json[i[0]]=coord_array
    return main_json

def coordinates_creator(x1,y1,x2,y2)->list:
    """
    Used to create convert x1,y1,x2,y2 to x1,y1,w,h format
    """
    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    h = round(y2-y1)
    w = round(x2-x1)
    x1 = round(x1)
    y1 = round(y1)
    return [x1,y1,w,h]

def json_arr_to_json(json_arr):
    main_json={}
    for json in json_arr:
        for key_ in json:
            main_json[key_]=json[key_]
    return main_json