import numpy as np
import cv2
import pandas as pd

def custom_rectangle(image, top_left, bottom_right, corner_length=5, thick_color=(0, 255, 0), thin_color=(0, 0, 255)):
    """
    Draws a rectangle with bright thick corners and a thin red line for all the boundaries.

    Parameters:
    image (numpy.ndarray): The image on which to draw the rectangle.
    top_left (tuple): The top-left coordinate of the rectangle (x, y).
    bottom_right (tuple): The bottom-right coordinate of the rectangle (x, y).
    corner_length (int): The length of the thick corners. Default is 20.
    thick_color (tuple): The color of the thick corners in BGR format. Default is white (255, 255, 255).
    thin_color (tuple): The color of the thin lines in BGR format. Default is red (0, 0, 255).

    Returns:
    numpy.ndarray: The image with the rectangle drawn on it.
    """
    corner_thinkness=3

    # Draw the thin red rectangle
    cv2.rectangle(image, top_left, bottom_right, thin_color, 1)

    # Draw the thick white corners
    # Top-left corner
    cv2.line(image, top_left, (top_left[0] + corner_length, top_left[1]), thick_color, corner_thinkness)
    cv2.line(image, top_left, (top_left[0], top_left[1] + corner_length), thick_color, corner_thinkness)

    # Top-right corner
    cv2.line(image, (bottom_right[0], top_left[1]), (bottom_right[0] - corner_length, top_left[1]), thick_color, corner_thinkness)
    cv2.line(image, (bottom_right[0], top_left[1]), (bottom_right[0], top_left[1] + corner_length), thick_color, corner_thinkness)

    # Bottom-left corner
    cv2.line(image, (top_left[0], bottom_right[1]), (top_left[0] + corner_length, bottom_right[1]), thick_color, corner_thinkness)
    cv2.line(image, (top_left[0], bottom_right[1]), (top_left[0], bottom_right[1] - corner_length), thick_color, corner_thinkness)

    # Bottom-right corner
    cv2.line(image, bottom_right, (bottom_right[0] - corner_length, bottom_right[1]), thick_color, corner_thinkness)
    cv2.line(image, bottom_right, (bottom_right[0], bottom_right[1] - corner_length), thick_color, corner_thinkness)

    return image

def plot_tracks(track,im:np.ndarray,debug=False)->np.ndarray:
        """
        Usage : Used to plot single tracks 
        Args  : 
            1. track (List) : List of coordinated and bbox related info
            2. im (np.ndarray) : image array
        Retruns : image with bbox im (np.ndarray) 
        """
        x3,y3,x4,y4,idx,obj_cls,confx,ind=track
        # if debug:
        #     print(f"obj_cls {obj_cls}")
        #     print( x3,y3,x4,y4,idx,confx,obj_cls,ind)
        x3=int(x3)
        y3=int(y3)
        x4=int(x4)
        y4=int(y4)
        idx=int(idx)

        #Plotting center point
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        bbox_list=[x3,y3,x4,y4]

        im=cv2.putText(im,f'ID - {idx}',(x3-10,y3-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
        
        # Turn the plotting on while debugging
        # if debug:
            # im=cv2.circle(im,(cx,cy),3,(255,0,0),3)
    
        # Define rectangle parameters
        top_left = (x3, y3)
        bottom_right = (x4, y4)
    
        # Draw the rectangle on the image
        im = custom_rectangle(im, top_left, bottom_right)
        return im,idx,cx,cy,bbox_list,obj_cls

def prediction_coordinated_hadler(results):
    """
    Provides coordinates for predictions done by YOLO
    """
    data = results[0].boxes.data
    data = data.detach().cpu().numpy()
    conf_=results[0].boxes.conf

    if conf_.nelement() != 0:
        conf=(results[0].boxes.conf[0].detach().cpu().numpy().astype("float"))
    
    else:
        conf=0
    
    px = pd.DataFrame(data).astype("float")
    
    return px,conf

def prediction_element_handler(px,conf):
    """
    Creates tracking element for and updates tracker list
    """
    
    temp_list=[]
    for index, row in px.iterrows():  
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            temp_list=[x1,y1,x2,y2,d,conf]
            
    return temp_list

def tracker_element_handler(dets,px,conf):
    """
    Creates tracking element for and updates tracker list
    """
    d=-1
    for index, row in px.iterrows():
            temp_list=[]
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            temp_list=[x1,y1,x2,y2,d,conf]
            dets.append(temp_list)
    return dets,d

# Function to plot lines with labels
def line_plotter(frame,line_name,x_start,x_end,y,line_color,text_color,line_thickness:int):
    frame=cv2.line(frame, (x_start, y), (x_end, y), line_color, line_thickness)
    frame=cv2.putText(frame, (str(line_name)), (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return frame



def finding_coordinates(px):
    temp_list=[]
    for index,row in px.iterrows():  
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            temp_list=[x1,y1,x2,y2]
    return temp_list



# Mouse callback function
def rectangle_coordinates_calculator(img, ix, iy, fx, fy,):
        """Plotting utils"""
        coordinates_json={}
        x_start,y_start,x_end,y_end=inner_rectangle_coordinates_calculator(ix, iy, fx, fy)
        coordinates_json['initial_x']=ix
        coordinates_json['initial_y']=iy
        coordinates_json['final_x']=fx
        coordinates_json['final_y']=fy
        coordinates_json['inner_x_start']=x_start
        coordinates_json['inner_y_start']=y_start
        coordinates_json['inner_x_end']=x_end
        coordinates_json['inner_y_end']=y_end
        return coordinates_json

# Function to draw the inner rectangle centered within the outer rectangle
def inner_rectangle_coordinates_calculator(start_x, start_y, end_x, end_y,length=82):
    """Plotting utils"""
    
    # Calculate the width of the outer rectangle
    width = abs(end_x - start_x)
    
    # Calculate the center position of the outer rectangle
    center_x = (start_x + end_x) // 2
    center_y = (start_y + end_y) // 2
    
    # Calculate the coordinates for the inner rectangle
    inner_start_x = start_x
    inner_start_y = center_y - length // 2
    inner_end_x = end_x
    inner_end_y = center_y + length // 2

    # Ensure the inner rectangle stays within the bounds of the outer rectangle
    if inner_start_y < start_y:
        inner_start_y = start_y
        inner_end_y = start_y + length
    if inner_end_y > end_y:
        inner_end_y = end_y
        inner_start_y = end_y - length

    return inner_start_x,inner_start_y,inner_end_x,inner_end_y

def rectangle_plotter(img,coordinates_json):
    """Plotting utils""" 
    img=cv2.rectangle(img, (coordinates_json['initial_x'], coordinates_json['initial_y']), (coordinates_json['final_x'], coordinates_json['final_y']), (0, 255, 0), 2)
    img=cv2.rectangle(img, (coordinates_json['inner_x_start'],coordinates_json['inner_y_start']), (coordinates_json['inner_x_end'], coordinates_json['inner_y_end']), (255, 0, 0), 2)
    return img
