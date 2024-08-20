import numpy as np
import easyocr
import pytesseract
from utils.checker_utils import remove_non_alphanumeric
from utils.tr_ocr_utils import eval_new_data

### Sequential_function


def easy_ocr_reader_sequential(image:np.ndarray)->str:
    """anrp utils"""
    ocr_text=[]
    # enhanced_img=enhance_image(image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
        if prob > 0:
            ocr_text.append(text)
    if ocr_text!=[]:
       return ocr_text[0]
    else:
       return '' 


def tesserac_ocr_reader_sequential(image:np.ndarray)->str:
    "anrp utils"
    text = pytesseract.image_to_string(image,lang='eng')
    return text

def trocr_processor_sequential(img, model, processor, device='cuda'):
    """
    anrp utils
    Process the image using a preloaded model and processor.
    """
    ocr_text = eval_new_data(image=img, model=model, processor=processor, device=device)
    return ocr_text 



def run_ocr(image:np.ndarray,tr_model,tr_device,tr_processor)->str:
    """
    anrp utils
    """
    es_ocr_result=easy_ocr_reader_sequential(image)
    tes_ocr_result=tesserac_ocr_reader_sequential(image)
    tr_ocr_result=trocr_processor_sequential(image,model=tr_model,device=tr_device,processor=tr_processor)
    return [remove_non_alphanumeric(es_ocr_result),remove_non_alphanumeric(tes_ocr_result),remove_non_alphanumeric(tr_ocr_result)]