import multiprocessing
import threading
import numpy as np
import easyocr
import pytesseract

from utils.tr_ocr_utils import eval_new_data


def mt_easy_ocr_reader(image:np.ndarray,queue=None)->str:
    """
    anrp utils
    """
    ocr_text=[]
    # enhanced_img=enhance_image(image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
        if prob > 0:
            ocr_text.append(text)
    if ocr_text!=[]:
        queue.put(ocr_text[0])
    else:
        queue.put("")

def mt_tesserac_ocr_reader(image:np.ndarray,queue=None)->str:
    """
    anrp utils
    """
    text = pytesseract.image_to_string(image,lang='eng')
    queue.put(text)
    
def mt_trocr_processor(img, model, processor,queue,device='cuda'):
    """
    anrp utils
    
    Process the image using a preloaded model and processor.
    """
    ocr_text = eval_new_data(image=img, model=model, processor=processor, device=device)
    queue.put(ocr_text)


def run_ocr_mt(image:np.ndarray,tr_model,tr_device,tr_processor)->str:
    """
    anrp utils
    """

    main_queue=multiprocessing.Queue()
    strings_=[]
    
    t1=threading.Thread(target=mt_easy_ocr_reader  ,args=[image,main_queue])
    t2=threading.Thread(target=mt_tesserac_ocr_reader,args=[image,main_queue])
    t3=threading.Thread(target=mt_trocr_processor,args=[image,tr_model,tr_device,tr_processor,main_queue])
    
    t1.start()
    t2.start()
    t3.start()
    
    t1.join()
    t2.join()
    t3.join()

    while not main_queue.empty():
        strings_.append(main_queue.get())
    
    return strings_