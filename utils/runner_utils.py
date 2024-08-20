import cv2
import numpy as np
import torch
import time

from ultralytics import YOLO

from utils.back_all_utils import json_to_csv, speed_calculator_kmph
from utils.blurring_utils import blur_except_rectangle
from utils.checker_utils import json_frame_order_checker
from utils.conversion_utils import  csv_to_json_arr, object_class_name_normalizer, violation_based_json_creator
from utils.date_time_utils import get_current_date, video_time_checker
from utils.file_utils import create_directory, detection_coordinate_write, evidance_directories_creator, evidance_img_separator
from utils.image_deblur_utils import deblur_images
from utils.json_utils import final_json_processor, ocr_result_filter, remove_empty_from_dict, voilation_capture_json_creator
from utils.model_utils import remove_model_from_gpu
from utils.mt_anrp_utils import main_process_detecting_and_cropping_liscense_plate, main_process_ocr_scanner_for_liscense_plate
from utils.mt_image_enhancement import main_process_image_enhance_using_dnn_supress_cpu
from utils.mt_image_utils import main_process_crop_images_using_paths, main_process_threshold_image_creator_using_paths
from utils.mt_video_utils import main_process_video_clip_creator
from utils.plotting_utils import line_plotter, plot_tracks, prediction_coordinated_hadler, tracker_element_handler
from utils.rectangle_utils import draw_rectangle_on_first_frame
from utils.tracker_utils import tracker_init
from utils.image_utils import blurred_frame_writer, frame_writer

def run_detection(video_file_path,camera_id,area_id,vehicle_json_arr:list,violation_json_arr:list,area_selection=False,debug:bool=False,gpu_device_number=1):
    down = {}
    up = {}
    tracking_area_coordinates=None
    text_color = (0, 0, 0)  # Black color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines


    counter_down = []
    counter_up = []


    offset = 6
    # Specify the start and end points of the line
    start_point = (300, 198)
    end_point = (300, 280)
    
    outer_x_start=321
    outer_x_end=801
    outer_y_start=129
    outer_y_end=416
    
    inner_x_start=321
    inner_x_end=801
    inner_y_start=231
    inner_y_end=313
  
    video_width=1020
    video_height=500
    frame_count=0
    
    all_id_tracker=[]
    voilation_id_tracker=[]
    violation_frame_tracker={}
    main_violation_tracker_json={}
    if area_selection:
        tracking_area_coordinates= draw_rectangle_on_first_frame(video_file_path, video_width, video_height)
        print(f"tracking_area_coordinates--{tracking_area_coordinates}")
        if tracking_area_coordinates:
            print("entered if-")
            outer_x_start=tracking_area_coordinates[0]['x_start']
            outer_x_end=tracking_area_coordinates[0]['x_end']
            outer_y_start=tracking_area_coordinates[0]['y_start']
            outer_y_end=tracking_area_coordinates[0]['y_end']
            inner_x_start=tracking_area_coordinates[1]['x_start']
            inner_x_end=tracking_area_coordinates[1]['x_end']
            inner_y_start=tracking_area_coordinates[1]['y_start']
            inner_y_end=tracking_area_coordinates[1]['y_end']
    

        
    
    vehicle_class_json=violation_based_json_creator(vehicle_json_arr)
    violation_type_json=violation_based_json_creator(violation_json_arr)

    tracker=tracker_init(cuda_device=torch.cuda.is_available(),cuda_device_number=gpu_device_number)
    # model=YOLO('models/custom_vehicle_detection.pt')
    model=YOLO('models/yolov8n.pt')
    
    yolo_model_classes=model.names
  
    
    cap = cv2.VideoCapture(video_file_path)
    assert cap.isOpened(), "Error reading video file"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_main = cv2.VideoWriter('output_strongs.avi', fourcc, 20.0, (1020, 500))
    all_frames_record_path=create_directory('all_frames_record')
    violation_frames_record_path=create_directory('all_violation_record')
    current_date=get_current_date()

    if debug:
        print("Starting video inference")
    
    while True:    
        ret, frame = cap.read()
        
        if not ret:
            break
        
        resized_image=cv2.resize(frame,(video_width, video_height))
        blurred_image=blur_except_rectangle(resized_image,outer_x_start,outer_y_start,outer_x_end,outer_y_end,offset_size=[0,80],blur_kernel_size=(50,50))
        
        # results=model.predict(blurred_image,conf=0.4,verbose=False,device=[gpu_device_number],classes=[2,3])
        results=model.predict(blurred_image,conf=0.4,verbose=False,device=[0],classes=[2,3])
        # print(f"\n----------------------------------------------------------------------------------\nResult without detection \n{results[0].boxes.data}\n----------------------------------------------------------------------------------\n")
        blurred_image=line_plotter(frame=blurred_image,x_start=inner_x_start,x_end=inner_x_end,y=inner_y_start,line_color=red_color,line_name='boundry 1',line_thickness=2,text_color=text_color)
        blurred_image=line_plotter(frame=blurred_image,x_start=inner_x_start,x_end=inner_x_end,y=inner_y_end,line_color=blue_color,line_name='boundry 2',line_thickness=2,text_color=text_color)
        
        # Specify the color of the line in BGR format (here, it's white)
        color = (255, 255, 255)
        
        # Draw the vertical line on the image
        thickness = 2  # You can adjust the thickness as needed
        cv2.line(blurred_image, start_point, end_point, color, thickness)
        
        # im=cv2.rectangle(blurred_image, (300, 198), (800, 280), (255, 255, 255), 2)
        
        px,conf=prediction_coordinated_hadler(results)
        dets = []
        # Experimenting
        dets,object_class_id=tracker_element_handler(dets,px,conf)        
        
        # print('out_from')
        
        dets = np.array(dets)
        # print(dets)
        if len(dets) > 0:
            tracks = tracker.update(dets, blurred_image) # --> M X (x, y, x, y, id, conf, cls, ind)
            # print('tracks',tracks)    
            for track in tracks:
                    travel_direction=-1
                    
                    blurred_image,object_id,cx,cy,bbox_list,object_class=plot_tracks(track,blurred_image,debug=True)
                    
                    object_class=int(object_class)

                    if f'{object_id}' in main_violation_tracker_json:
                        pass
                    else:
                        main_violation_tracker_json[f'{object_id}']={}
                        #Remember by adding this we are trying to handle multiple violations in a single array
                        #We also need to improvde the checker utils and 
                        main_violation_tracker_json[f'{object_id}']['violation_type_id']=[]
                        # print("adding Id to tracker")
                    all_id_tracker.append(object_id)
                    
                    # Adding file path for csv
                    all_detections_track_csv=detection_coordinate_write(frame_count,object_id,bbox_list,object_class,f'{current_date}_all_frame_detection_detail.csv')
                
                    if inner_y_start<(cy+offset) and inner_y_start > (cy-offset):
                        # print("entered if 1")
                        down[object_id]=time.time()
                
                    if object_id in down:
                        # print("entered if 2")
                
                        if inner_y_end<(cy+offset) and inner_y_end > (cy-offset):
                            # print("entered if 3")
                            travel_direction=1
                            elapsed_time=time.time() - down[object_id] 
                            # print("entered if 4")
                            if counter_down.count(object_id)==0:
                                counter_down.append(object_id) 
                                distance = 9
                                est_speed=speed_calculator_kmph(distance,elapsed_time)
                                if est_speed>20:
                                    
                                    # Fucntion for timer tracking
                                    time_dict=video_time_checker(count=frame_count,cap=cap)
                                    
                                    # Getting the name for the detected object
                                    object_name=object_class_name_normalizer(object_class_id,yolo_model_classes)
                                    
                                    main_violation_tracker_json[f'{object_id}']['violation_type_id'].append(violation_type_json['over_speed'])
                                    main_violation_tracker_json[f'{object_id}']['vehicle_type_id']=vehicle_class_json[f'{object_name}']
                                    
                                    main_violation_tracker_json[f'{object_id}']['vehicle_speed']=est_speed
                                    main_violation_tracker_json[f'{object_id}']['travel_direction']=travel_direction

                                    main_violation_tracker_json[f'{object_id}']['location_id']=area_id
                                    main_violation_tracker_json[f'{object_id}']['camera_id']=camera_id

                                    main_violation_tracker_json[f'{object_id}']['violation_date']=current_date
                                    main_violation_tracker_json[f'{object_id}']['violation_time']=time_dict['time']
                                    
                                    voilation_id_tracker.append(object_id)
                                    violation_frame_tracker[object_id]=frame_count
                                    
                                    violation_frame_writer_original=frame_writer(frame_count,resized_image,violation_frames_record_path)
                                    violation_frame_writer_blurred=blurred_frame_writer(frame_count,resized_image,violation_frames_record_path,bbox_list=bbox_list)
                                    
                                    main_violation_tracker_json[f'{object_id}']['evidence_focused_image_url']=violation_frame_writer_blurred
                                    main_violation_tracker_json[f'{object_id}']['evidence_image_url']=violation_frame_writer_original
                                                
                                    violation_csv_file_path=detection_coordinate_write(frame_count,object_id,bbox_list,object_class,f'{current_date}_violation_frame_detection_detail.csv')
                                    
                                # print(f"for {object_id} speed is -> {est_speed}")
        
        _=frame_writer(frame_count,resized_image,all_frames_record_path)    
        frame_count+=1
        out_main.write(resized_image)
        
        # # break on pressing q or spaceq
        cv2.imshow('Strong Sort Detection', blurred_image)     
        # key = cv2.waitKey(25) & 0xFF
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    remove_model_from_gpu(model)
    if debug:
        print("ending video inference")
    print(violation_frame_tracker)
    main_violation_json=voilation_capture_json_creator(voilation_id_tracker,all_detections_track_csv)
    maintained=json_frame_order_checker(main_violation_json)
    if debug:
        print(f"Json format is maintained {maintained}")
    if maintained:
        # Function for creating dir paths only
        evidance_img_dir_paths,evidance_clip_dir_paths=evidance_directories_creator(main_violation_json)
        if debug:
            print("Debug Process 1 Completed")
        # Function for taking separating image
        ### This should also add vehicle class name while making the folders 
        evidance_img_dir_paths=evidance_img_separator(evidance_img_dir_paths,all_frames_record_path,main_violation_json)   
        
        if debug:
            print("Debug Process 2 Completed")
        
        violation_coordinates_json=csv_to_json_arr(file_path=all_detections_track_csv,violation_json=violation_frame_tracker)
        
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        main_process_video_clip_creator(voilation_id_tracker,violation_frame_tracker,evidance_clip_dir_paths,all_frames_record_path,violation_coordinates_json)
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################

        if debug:
            print("Debug Process 3 Completed")
        
        cropped_images_path_list=main_process_crop_images_using_paths(evidance_img_dir_paths,main_violation_json)
        if debug:
                print("Debug Process 4 Completed")
        
        deblurred_image_paths=deblur_images(cropped_images_path_list,main_violation_json)
        if debug:
            print("Debug Process 6 Completed")
        
        # Using default 8 threads to achive max effiencicy
        enhance_image_path=main_process_image_enhance_using_dnn_supress_cpu(deblurred_image_paths,num_threads=8)
        if debug:
            print("Debug Process 7 Completed")
        
        cropped_lp_paths=main_process_detecting_and_cropping_liscense_plate(enhance_image_path,'models/LPD.pt',2)
        if debug:
            print("Debug Process 8 Completed")
        
        enhanced_cropped_images_lp=main_process_image_enhance_using_dnn_supress_cpu(cropped_lp_paths,num_threads=8,dir_name='cropped_lp_enhanced')
        if debug:
            print("Debug Process 9 Completed")
        
        thresholded_image=main_process_threshold_image_creator_using_paths(enhanced_cropped_images_lp)
        if debug:
            print("Debug Process 10 Completed")
        
        lp_csv_path=main_process_ocr_scanner_for_liscense_plate(folder_paths=thresholded_image,threads=1,gpu_device_number=1)
        if debug:
            print("Debug Process 11 Completed")
        
        ocr_json=ocr_result_filter(main_violation_tracker_json,lp_csv_path)
        if debug:
            print("Debug Process 12 Completed")
        
        main_violation_tracker_json=final_json_processor(main_violation_tracker_json,ocr_json)
        if debug:
            print("Debug Process 13 Completed")
        main_violation_tracker_json=remove_empty_from_dict(main_violation_tracker_json)
        if debug:
            print("Debug Process 14 Completed")
        csv_file_name=f'{get_current_date()}_inference_results.csv'
        results_csv_path=json_to_csv(json_data=main_violation_tracker_json,csv_filename=f'{csv_file_name}',debug=True)
        return results_csv_path 