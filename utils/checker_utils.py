import re
from collections import Counter


def frame_order_checker(arr):
    """
    checker_utils

    Used to check the order of frames appended in the violation json
    """
    order_maintained=True
    for cnt,i in enumerate(arr):
        if str(cnt+1) >= str(len(arr)):
            # print("finished")
            break
        if int(arr[cnt][0]) <= int(arr[cnt+1][0]):
            continue
        else:
            order_maintained=False
    return order_maintained

def json_frame_order_checker(main_json,debug=False):
    """
    Used to check the order of frames
    """
    maintainer=True
    for keys_ in main_json:
        order_maintained=frame_order_checker(main_json[keys_])
        if debug:
            print(f"order_maintainer - {order_maintained}")
        if not order_maintained:
            maintainer=False
    return maintainer

def compare_strings(strings):
    if not strings:
        return ""
    # Find the maximum length of the strings
    max_length = max(len(s) for s in strings)
    final_string = []
    for i in range(max_length):
        # Collect all crereharacters at the current position from all strings
        chars_at_pos = [s[i] for s in strings if i < len(s)]
        
        # Find the most common character at this position
        if chars_at_pos:
            most_common_char, _ = Counter(chars_at_pos).most_common(1)[0]
            final_string.append(most_common_char)
        else:
            final_string.append(' ')
    
    return ''.join(final_string)  

def valid_license_plate(plate):
    """
    Validates if the given string is in the format of an Indian license plate.
    
    Args:
    plate (str): The license plate string to validate.
    
    Returns:
    bool: True if the plate is valid, False otherwise.
    """
    # Regular expression to match the common Indian license plate format
    pattern = r'^[A-Z]{2}\d{2}[A-Z\d]{1,4}$'
    
    # Using fullmatch to ensure the entire string matches the pattern
    return bool(re.fullmatch(pattern, plate))

def remove_non_alphanumeric(input_string):
    """checker utils"""
    pattern = r'[^a-zA-Z0-9]+'
    result = re.sub(pattern, '', input_string)
    return result
