

import cv2
import queue
import numpy as np
import time
import threading
import lcd1602
from onnxruntime_infer.rapid_ocr_api import TextSystem
from picamera2.picamera2 import Picamera2, Preview

def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cropImg(img, boxes, size):
    x,y,w,h = boxes
    crop_img = img[y:y+h, x:x+w]
    return crop_img



def setup():
    ''' 
        load yolov4 and rapidOCR.
        do not modified this function.
    ''' 
    weight_path = r"./yolov4/yolov4-tiny-obj_best.weights"
    cfg_path = r"./yolov4/yolov4-tiny-obj.cfg"

    det_model_path = './onnxruntime_infer/models/ch_ppocr_mobile_v2.0_det_infer.onnx'
    cls_model_path = './onnxruntime_infer/models/ch_ppocr_mobile_v2.0_cls_infer.onnx'

    rec_model_path = './onnxruntime_infer/models/en_number_mobile_v2.0_rec_infer.onnx'
    keys_path = './onnxruntime_infer/rec_dict/en_dict.txt'

    net = cv2.dnn.readNet(weight_path, cfg_path)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    text_sys = TextSystem(det_model_path,
                        rec_model_path,
                        use_angle_cls=True,
                        cls_model_path=cls_model_path,
                        keys_path=keys_path)
    return model, text_sys


def rapidOCR(name, box, text_sys):
    dt_boxes, rec_res = text_sys(name, box)
    return rec_res


def create_boxes(bshape):
    x, y = bshape[1] ,bshape[0]
    box = np.empty((1, 4, 2))
    box[0][0] = np.array([0., 0.])
    box[0][1] = np.array([x, 0])
    box[0][2] = np.array([x, y])
    box[0][3] = np.array([0, y])
    return box

def text_postProcess(text):
    newT = ""
    # print("post",text)
    # print("post",len(text))
    if len(text) <= 4:         #remove the case which detect fail 
        return newT 
    for i in range(len(text)):
        # print("post",text[i])
        if text[i].isalpha() or text[i].isdigit():
            newT += text[i]
    return newT

def ALPR(model, text_sys, img, lcd, cnt):
    global can_put
    size = img.shape[:2]
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA) # reszie to smaller pic increase processing speed
    res = model.detect(img, 0.3, 0.2) #return object position 
    try: # do below if a license is detected
        classes, confidence, boxes = res
        classes, confidence, boxes = classes[0], confidence[0], boxes[0]
        crop_img = cropImg(img, boxes, size)
        box = []
    except: # otherwise return
        print("could not detect number plate")
        return
    can_put=False #if the model is doing prediction, putting frames into queue is not allow
    name = './cropped/cropped' + str(cnt) + '.jpg'
    #show_img('croped', crop_img)
    cv2.imwrite(name, crop_img)
    res = rapidOCR(name, box, text_sys) # feed the license plate pic to a text recognition model
    try:
        res = sorted(res, key=lambda tup: tup[1]) # get the result with highest probability one
        text = text_postProcess(res[0][0]) #keep number and alphabet, remove sign
        print(text)
        lcd.text(text, 1) #show result on lcd
    except:
        print(res)
    can_put=True #allow


def process(model, text_sys, lcd):
    global q
    cnt = 0
    while True:
        if q.empty() != True:
            img = q.get() # get a frame from queue
            cnt+=1
            ALPR(model, text_sys, img, lcd, cnt)
        
def webcam_show():
    global q
    global can_put
    can_put=True
    q = queue.Queue() # a queue that store some frames to be recognized
    cnt = 0
    cap = cv2.VideoCapture(-1)
    while True:
        cnt+=1
        ret, img = cap.read() #a frame catched from webcam
        
        if cnt >= 45 and can_put: #put one frame into queue evrey 45 frames
            q.put(img)
            cnt = 0
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('show_frame.jpg', img)
        k = cv2.waitKey(1)
        if k != -1:
            break
    cap.release()
    cv2.destroyAllWindows()

def picam_show():
    global q
    global can_put
    can_put=True
    q = queue.Queue() # a queue that store some frames to be recognized
    cnt = 0
    picam2 = Picamera2()
    picam2.configure(picam2.preview_configuration(main={"format": "RGB888", "size": (1024,768)}))
    picam2.start()
    while True:
        cnt+=1
        img = picam2.capture_array() #a frame catched from picamera
        
        if cnt >= 45 and can_put: #put one frame into queue evrey 45 frames
            q.put(img)
            cnt = 0
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('show_frame.jpg', img)
        k = cv2.waitKey(1)
        if k != -1:
            break
    picam2.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        lcd = lcd1602.LCD_init()
        lcd.text("LPR system!", 1)
    except:
        print("lcd not detected")

    model, text_sys = setup()
    t = threading.Thread(target=webcam_show) #target= webcam_show or picam_show
    t1 = threading.Thread(target=process, args=(model, text_sys, lcd))
    t.start()
    t1.start()   
    


