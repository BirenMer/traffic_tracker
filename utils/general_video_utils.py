import os
import cv2
from utils.blurring_utils import blur_except_rectangle
from utils.conversion_utils import arr_to_coordinate_json,coordinates_creator
def video_clip_creator(input_file_list,output_file_name='evidance_x',output_folder_path='evidance_clips',coordinates_arr=None):
    video_file_path_path=os.path.join(output_folder_path,f'{output_file_name}.avi')
    object_id=video_file_path_path.split('/')[-1]
    # print(f'From video clip creator - {i}')
    coord_json=arr_to_coordinate_json(coordinates_arr)
    print(coordinates_arr)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{video_file_path_path}', fourcc, 20.0, (1020, 500))
    for i in input_file_list:
        # print(f'From violation - {i}')
        frame = cv2.imread(i)
        frame_number=str(os.path.basename(i).split('.')[0])
        coordinates=coord_json[frame_number]
        x1,y1,x2,y2=coordinates
        x1,y1,w,h=coordinates_creator(x1,y1,x2,y2)
        ### Wee need bluring funciton here
        frame=blur_except_rectangle(image=frame,x=x1,y=y1,width=w,height=h)
        ### Wee need tracking utils here
        out.write(frame)
    out.release()
    return video_file_path_path
