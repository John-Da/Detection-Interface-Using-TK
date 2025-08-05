from ultralytics import YOLO

# from util.popupmsg import message_box

def load_model(weights_folder, selected_model):
    chose = f"{weights_folder}/{selected_model.get()}"
    model = YOLO(chose)
    return model
        

def run_model(model, image_cv, myconf=0.25):
    if image_cv is None:
        print("No image data to process")
        return None
    result = model(image_cv, conf=myconf, save=False, verbose=False)
    
    return list(result)
