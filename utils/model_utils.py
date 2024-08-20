import torch
import gc

from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def remove_model_from_gpu(model_object):
    del model_object
    gc.collect()
    torch.cuda.empty_cache() 
    return True

def YOLO_model_loader(model_path,num_threads):
    all_models=[]
    for i in range(0,num_threads):
        model=YOLO(model_path)
        all_models.append(model)
    return all_models

def tr_ocr_model_loader(numthreads=3,cuda_on=True,model_name='microsoft/trocr-small-printed',device_number=1):
    """
    Model utils
    """
    if cuda_on:
        device = f'cuda:{device_number}' if torch.cuda.is_available() else 'cpu'
    else:
        device='cpu'
    model_json_list=[]
    for _ in range(numthreads):
        model_dict={}
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        processor = TrOCRProcessor.from_pretrained(model_name)
        # Loading params into temp json
        model_dict['device']=device
        model_dict['model_name']=model_name
        model_dict['model']=model
        model_dict['processor']=processor
        model_json_list.append(model_dict)
        print(f"Done_loading {_}")
    return model_json_list


def tr_ocr_model_deloader(model_json_list):
    """
    Model utils
    """
    for model_dict in model_json_list:
        # Move model to CPU to free GPU memory
        model_dict['model'].cpu()
        # Remove references to model and processor
        del model_dict['model']
        del model_dict['processor']
    
    # Clear the cache and run garbage collector
    torch.cuda.empty_cache()
    gc.collect()