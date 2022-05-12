
from venv import create
import cv2
from PIL import Image
import pytesseract
import numpy as np
import time
from onnxruntime_infer.rapid_ocr_api import TextSystem

def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cropImg(img, boxes, size):
    #(H, W) = size
    #h_s = H/416
    #w_s = W/416
    x,y,w,h = boxes
    crop_img = img[y:y+h, x:x+w]
    #mx = max(crop_img.shape[:2])
    #scale = 300/mx
    #crop_img = cv2.resize(crop_img, None, fx=scale*w_s, fy=scale*h_s)
    return crop_img



def setup():
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

def ALPR(model, text_sys, image_path):
    img = cv2.imread(image_path)
    size = img.shape[:2]
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    res = model.detect(img, 0.3, 0.2)
    if len(res) == 3:
        classes, confidence, boxes = res
        classes, confidence, boxes = classes[0], confidence[0], boxes[0]
        crop_img = cropImg(img, boxes, size)
        box = []
    else:
        print("could not detect number plate")
        return
        
    name = 'croped.jpg'
    #show_img('croped', crop_img)
    cv2.imwrite(name, crop_img)
    print(rapidOCR(name, box, text_sys))

if __name__ == '__main__':
    model, text_sys = setup()
    img_foloer = './images/'
    img_list = ['car.jpg', 'car4.jpg', 'test.jpg']
    for i in range(3):
        ALPR(model, text_sys, (img_foloer+img_list[i]))


