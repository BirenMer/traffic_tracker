def tr_ocr(image, processor, model, generation_length, device):
    """anrp utils"""
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=generation_length)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def eval_new_data(image=None, model=None, processor=None, device='cpu'):
    """anrp utils"""
    generation_length = 12
    text = tr_ocr(image, processor, model, generation_length, device)
    return text
