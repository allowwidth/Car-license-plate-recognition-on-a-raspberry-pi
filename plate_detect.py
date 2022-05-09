import cv2 as cv
import numpy as np
path = './plate.jpg'


def show_img(name, img, wait=False):
    cv.imshow(name ,img)
    # if wait:
    cv.waitKey(0)
    cv.destroyAllWindows()

def find_license(img_canny, img):
    contours, hier = cv.findContours(img_canny.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    img_copy = img.copy()
    show_img('test', img)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:50]
    cv.drawContours(img_copy, contours, -1, (0, 255, 0), 3)
    show_img('line', img_copy)

if __name__ == '__main__':
    img = cv.imread(path)
    if img == [] :
        print('Read image failed!!')
    else :
        print(img.shape)
    # img = cv.resize(img, (600, 400))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 5, 15, 15)
    # show_img('gray', gray)
    edge = cv.Canny(gray, 30, 150)
    # show_img('canny', edge, True)
    find_license(edge, img)


