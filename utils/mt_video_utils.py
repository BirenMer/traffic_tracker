import threading

from utils.conversion_utils import json_arr_to_json
from utils.file_utils import frame_picker
from utils.general_video_utils import video_clip_creator

def sub_process_video_clip_creator(entry,violation_frame_tracker,all_frames_record_path,window_size=20,violation_coord_json=None):
        """
        Video clip utils
        """
        print("PRitning from video clips")
        detection_json=json_arr_to_json(violation_coord_json)
        previous_frames,post_frames=frame_picker(violation_frame_tracker[entry[0]],all_frames_record_path,window_size)

        input_frame_list=previous_frames+post_frames
        
        video_clip_creator(input_frame_list,f'evidance_{entry[0]}',output_folder_path=entry[2],coordinates_arr=detection_json[str(entry[0])])

def main_process_video_clip_creator(voilation_id_tracker,violation_frame_tracker,evidance_clip_dir_paths,all_frames_record_path,violation_coord_json):
    """
    video_utils
    Used to create and store video clips from the give frame paths to the given path
    """
    thread_list_=[]

    for i in zip(voilation_id_tracker,violation_frame_tracker,evidance_clip_dir_paths):    
        t=threading.Thread(target=sub_process_video_clip_creator,args=[i,violation_frame_tracker,all_frames_record_path,20,violation_coord_json])
        t.start()
        thread_list_.append(t)

    for j in thread_list_:
        j.join()
    return evidance_clip_dir_paths