# %%
import cv2
from PIL import Image
import pytesseract
import easyocr

# %%
def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sharpen(img, sigma=100):    
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

def cropImg(img, boxes, size):
    (H, W) = size
    h_s = H/416
    w_s = W/416
    x,y,w,h = boxes
    crop_img = img[y:y+h, x:x+w]
    mx = max(crop_img.shape[:2])
    scale = 300/mx
    crop_img = cv2.resize(crop_img, None, fx=scale*w_s, fy=scale*h_s)
    return crop_img

def OCR(name):
    # crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    # # crop_img = Image.open('car.jpg')
    # text = pytesseract.image_to_string(crop_img, lang='eng')
    # print(text)
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(name)
    print(result[0][1])



# %%

weight_path = r"./yolov4/yolov4-tiny-obj_best.weights"
cfg_path = r"./yolov4/yolov4-tiny-obj.cfg"

# %%
net = cv2.dnn.readNet(weight_path, cfg_path)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# %%
def rapidOCR():
    from onnxruntime_infer.rapid_ocr_api import TextSystem, visualize

    det_model_path = './models/ch_ppocr_mobile_v2.0_det_infer.onnx'
    cls_model_path = './models/ch_ppocr_mobile_v2.0_cls_infer.onnx'

    # 中英文识别
    rec_model_path = 'models/ch_ppocr_mobile_v2.0_rec_infer.onnx'
    keys_path = 'rec_dict/ppocr_keys_v1.txt'

    text_sys = TextSystem(det_model_path,
                        rec_model_path,
                        use_angle_cls=True,
                        cls_model_path=cls_model_path,
                        keys_path=keys_path)

    image_path = r'test_images/det_images/ch_en_num.jpg'
    dt_boxes, rec_res = text_sys(image_path)

# %%


img_foloer = './images/'
img_list = ['car.jpg', 'car1.jpg', 'test.jpg']
for i in range(1):
    img = cv2.imread(img_foloer+img_list[i])
    print(img.shape)
    size = img.shape[:2]
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    res = model.detect(img, 0.3, 0.2)
    if len(res) == 3:
        classes, confidence, [boxes] = res
        crop_img = cropImg(img, boxes, size)
    else:
        print("could not detect number plate")
        break
    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # ret, crop_img = cv2.threshold(crop_img, 100, 255, cv2.THRESH_BINARY_INV)
    name = 'croped.jpg'
    #show_img('croped', crop_img)
    cv2.imwrite(name, crop_img)
    # OCR(name)


