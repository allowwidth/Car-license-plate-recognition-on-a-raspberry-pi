# %%
import cv2

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
    h_s = H/608
    w_s = W/608
    x,y,w,h = boxes
    crop_img = img[y:y+h, x:x+w]
    mx = max(crop_img.shape[:2])
    scale = 300/mx
    crop_img = cv2.resize(crop_img, None, fx=scale*w_s, fy=scale*h_s)
    return crop_img

# %%
absolute = r"/home/pi/Desktop/ES_final"
weight_path = r"/yolov4-tiny-obj_best.weights"
cfg_path = r"/yolov4-tiny-obj.cfg"

# %%
net = cv2.dnn.readNet(absolute+weight_path, absolute+cfg_path)

# %%


img = cv2.imread('car.jpg')
size = img.shape[:2]
img = cv2.resize(img, (608, 608), interpolation=cv2.INTER_AREA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255, swapRB=True)
classes, confidence, [boxes] = model.detect(img, 0.3, 0.2)
crop_img = cropImg(img, boxes, size)
# crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
# ret, crop_img = cv2.threshold(crop_img, 100, 255, cv2.THRESH_BINARY_INV)
# show_img('croped', crop_img)
# print(type(crop_img))
# cv2.imwrite('croped.jpg', crop_img)

# %%
from PIL import Image
import pytesseract

crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
# crop_img = Image.open('car.jpg')
text = pytesseract.image_to_string(crop_img, lang='eng')
print(text)

# %%
# import easyocr
# reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
# result = reader.readtext('croped.jpg')


