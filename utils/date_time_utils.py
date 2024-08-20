import cv2
from datetime import date
from utils.conversion_utils import convert_seconds_to_hhmmss


def get_current_date():
  """
  date_time_utils
  Returns the current date in DD-MM-YYYY format.
  """
  today = date.today()
  return today.strftime("%d_%m_%Y")


# Function to track time in video based on FPS and Frame count
def video_time_checker(count,cap):
        """
        date time utils"""
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000.0
        current_time_formatted = convert_seconds_to_hhmmss(current_time_sec)
        return {
                "frame":count,
                "time": current_time_formatted
               }