from pathlib import Path
from boxmot import StrongSORT
from boxmot import DeepOCSORT

def tracker_init(tracker_name:str='strong_sort',reid_model_path:str='models/osnet_x0_25_msmt17.pt',cuda_device:bool=True,cuda_device_number:int=0):
    """
    Usage: Used to initialize trackers || Default=strong_sort
               Currently support two Trackers :
               1. strong_sort (default)
               2. deep_oc_sort
    Args : 
    1. tracker_name(str) -> used to select a traker || Default = strong_sort
    2. reid_model_path(str) -> used to select a reid model  || Defualt = osnet_x0_25_msmt17
    3. cuda_device(bool) -> Used to select between CPU and GPU for calculation || Default=GPU
    4. cuda_device_number(int) -> Used to select GPU device (If there exists multiple device) || Default = 0

    Returns : Returns a tracker object 


    List of available reid model  : ['resnet50', 'resnet101', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 'hacnn', 'mlfn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0', 'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25', 'lmbn_n', 'clip']
    """
    tracker=None
    if tracker_name=='deep_oc_sort':
        tracker = DeepOCSORT(
        model_weights=Path(reid_model_path), # which ReID model to use
        device=f'cuda:{cuda_device_number}' if cuda_device else 'cpu',
        fp16=False,
        )

    else:
        tracker = StrongSORT(
        model_weights=Path(reid_model_path), # which ReID model to use
        device=f'cuda:{cuda_device_number}' if cuda_device else 'cpu',
        fp16=False,
        )
    return tracker
