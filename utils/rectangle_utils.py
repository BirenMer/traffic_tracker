import cv2
import numpy as np

# Global variables
ix, iy = -1, -1
fx, fy = -1, -1
drawing = False
img = None
inner_coords = None
outer_coords = None

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, img, inner_coords, outer_coords

    if event == cv2.EVENT_LBUTTONDOWN:
        # When the left mouse button is pressed, start drawing
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # If the mouse is moving while drawing, show the rectangle being drawn
            img_temp = img.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Select area and Press q', img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        # When the left mouse button is released, finish drawing
        drawing = False
        fx, fy = x, y
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 255), 2)
        y_start, y_end = draw_inner_rectangle(ix, iy, fx, fy)
        inner_coords = {
            "x_start": ix,
            "y_start": y_start,
            "x_end": fx,
            "y_end": y_end
        }
        outer_coords = {
            "x_start": ix,
            "y_start": iy,
            "x_end": fx,
            "y_end": fy
        }
        cv2.imshow('Select area and Press q', img)

# Function to draw the inner rectangle centered within the outer rectangle
def draw_inner_rectangle(start_x, start_y, end_x, end_y):
    global img
    # Calculate the width of the outer rectangle
    width = abs(end_x - start_x)
    # Define the fixed length for the inner rectangle
    length = 82

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

    # Draw the inner rectangle
    cv2.rectangle(img, (inner_start_x, inner_start_y), (inner_end_x, inner_end_y), (255, 0, 0), 2)
    return inner_start_y, inner_end_y

def draw_rectangle_on_first_frame(video_path, display_width=None, display_height=None):
    global img, inner_coords, outer_coords

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None

    # Read the first frame
    ret, img = cap.read()
    
    # Check if frame reading was successful
    if not ret or img is None:
        print("Error: Could not read frame.")
        return None, None

    # Resize the image if display dimensions are provided
    if display_width and display_height:
        img = cv2.resize(img, (display_width, display_height))

    # Create a window and set the mouse callback function
    cv2.namedWindow('Select area and Press q')
    cv2.setMouseCallback('Select area and Press q', draw_rectangle)

    while True:
        # Display the frame continuously until 'q' is pressed
        cv2.imshow('Select area and Press q', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Release the video capture object
    cap.release()
    if outer_coords and inner_coords:
        return [outer_coords, inner_coords]
    else:
        return None



