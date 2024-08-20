import threading
from utils.image_utils import *
from utils.file_utils import frame_picker
from utils.general_video_utils import video_clip_creator

def video_clip_creator_using_id(voilation_id_tracker,violation_frame_tracker,evidance_clip_dir_paths,all_frames_record_path):
    """
    Used to create and store video clips from the give frame paths to the given path
    """
    for i in zip(voilation_id_tracker,violation_frame_tracker,evidance_clip_dir_paths):    
        previous_frames,post_frames=frame_picker(violation_frame_tracker[i[0]],all_frames_record_path,20)
        input_frame_list=previous_frames+post_frames
        video_clip_creator(input_frame_list,f'evidance_{i[0]}',output_folder_path=i[2])
    return evidance_clip_dir_paths

