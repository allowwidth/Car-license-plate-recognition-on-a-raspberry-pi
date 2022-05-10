


import cv2
from picamera2.picamera2 import Picamera2
import time





def capture():
    picam2 = Picamera2()
    picam2.configure(picam2.preview_configuration(main={"format": "RGB888", "size": picam2.sensor_resolution}))
    picam2.start()
    time.sleep(1)
    im = picam2.capture_array()
    picam2.close()
    return im

if __name__ == '__main__':
    img = capture()
    cv2.imwrite('te.jpg', img)
    picam2.close()
    