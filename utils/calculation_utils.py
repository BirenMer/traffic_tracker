#Function to calculate Speed 
def speed_calculator_kmph(distance,elapsed_time):
    """
    calculation utils
    Args: distance : int , elasped_time: float
    Usage: To estimate the speed of the object.
    Return: speed_kmh : int (Speed in Kilometer per hour)
    """
    speed_kmh = (distance*2  / elapsed_time) * 3.6 # conveting this into km/h 
    return int(speed_kmh)