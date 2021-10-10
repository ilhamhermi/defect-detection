from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from imutils import build_montages
from imutils import paths
import numpy as np 
import matplotlib.pyplot as plt
import random
from cv2 import cv2
import serial
import time
import threading
from multiprocessing import Process
import sys
import imutils
from deskew import determine_skew

from alibi_detect.utils.saving import load_detector
from PIL import Image
import glob
import keras.optimizers
from alibi_detect.od import OutlierAE
from alibi_detect.utils.fetching import fetch_detector
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,\
    Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D

import math
from typing import Tuple, Union

data = ''
gambar = None
# model = load_model('/media/ilham/Data 3/ubuntu/Project_an/model2')
# cap = cv2.VideoCapture(1)

def main_function():
    global data
    # global model
    # global cap
    global ser
    data = 'n'
    # model = load_model('/media/ilham/Data 3/ubuntu/Project_an/saveModel.model')
    # od = load_detector('/media/ilham/Data 3/ubuntu/Project_an/dataset/detector9-20210409T142045Z-001/detector9')
    # od = load_detector("/media/ilham/Data 3/ubuntu/Project_an/detector-20210409T180846Z-001/detector")
    od = load_detector('/media/ilham/Data 3/ubuntu/Project_an/dataset/detector200p/content/detector200p')
    # od = load_detector("/media/ilham/Data 3/ubuntu/Project_an/detector200p/content/detector200p")
    # model = load_model('/media/ilham/Data 3/ubuntu/Project/dataset/mymodel-20210221T145745Z-001')
    cap = cv2.VideoCapture(1)
    focus = cv2.CAP_PROP_FOCUS
    cap.set(focus,20)
    ser = serial.Serial('/dev/ttyUSB0',baudrate=115200,timeout=1)
    # time.sleep(3)
    # ser.write('#'.encode())
    # time.sleep(3)
    # ser.write('@'.encode())
    # while True:
    # lock = threading.Lock()
    # t1 = threading.Thread(target=ko
    # munikasiSerial,args=(lock,ser,))
    # t2 = threading.Thread(target=ambilGambar,args=(lock,cap,model,))
    time.sleep(5)
    ser.write('S'.encode())
    # t1.setDaemon(True)
    # t2.setDaemon(True)

    # t1.start()
    # t2.start()

    # t1.join()
    # t2.join()

    while True:
        # pass
        ambilGambar(cap,od)
        komunikasiSerial(ser)
        # t1.join()
        # t2.join()

        # print(data)
        # _, img = cap.read()
        # cv2.imshow("gambar", img)
        # data = ser.read(1).decode('ascii')
        # # print(data)
        # k = cv2.waitKey(1)&0xFF

        # if k == 27:
        #     cap.release()
        #     cv2.destroyAllWindows()
            
        # # if k == ord('s'):
        # if(data == 't'):
        #     gambar = "dataset/testing/bahan.png"
        #     cv2.imwrite(gambar,img)

        #     bahan = image.load_img(gambar,target_size=(200,200))
        #     x = image.img_to_array(bahan)
        #     x = np.expand_dims(x,axis=0)

        #     images = np.vstack([x])
        #     val = model.predict(images)

        #     if val == 0:
        #         print("DEFECT")
        #         ser.write('#'.encode())
        #     else:
        #         print("NORMAL")
        #         ser.write('@'.encode())

def ambilGambar(cap, model):
    global data
    _, img = cap.read()
    cv2.imshow("gambar", img)
    k = cv2.waitKey(1)&0xFF

    if k == 27:
        ser.write('!'.encode())
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
           
    if(data == 't'):
        lokasiGambar = "dataset/testing/bahan.png"
        print("take")
        rot = putar(img)
        crop = cropGambar(rot)
        cv2.imwrite(lokasiGambar,crop)
        detect(lokasiGambar,model)
                
def komunikasiSerial(serial):
    # global ser
    global data
    # while True:
        # lock.acquire()
    if(ser.in_waiting):
        data = serial.read(1).decode('ascii')
        print("serial :",data)
        # lock.release()
        # time.sleep(1/1000000000)

def cropGambar(image):
    global gambar

    output = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    circles  = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.2,400)
    # cv2.imshow("gray", circles)
    if circles is not None:
        
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            miny=y-r-7
            minx=x-r-7
            maxy=y+r+7
            maxx=x+r+7
        
        if miny < 0:
            miny = 0
        if minx < 0:
            minx = 0
        if maxx > output.shape[1]:
            maxx = output.shape[1]
        if maxy > output.shape[0]:
            maxy = output.shape[0] 
        output = output[miny:maxy,minx:maxx]
        print("berhasil")
        # gambar = cv2.resize(gambar,(200,200))
        return output
    else:
        print("gagal")
    # gambar = output
    
    # gambar = cv2.resize(gambar,(200,200))
def img_to_np(path, resize = True):  
    img_array = []
    fpaths = glob.glob(path, recursive=True)
    for fname in fpaths:
        img = Image.open(fname).convert("RGB")
        if(resize): img = img.resize((200,200))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images
def detect(pathImage,od):
    test = img_to_np(pathImage)
    test = test.astype('float32') / 255.
    od.infer_threshold(test,threshold_perc=98)
    preds = od.predict(test, outlier_type='instance',
            return_instance_score=True,
            return_feature_score=True)
    print(preds['data']['is_outlier'][0])
    print(preds['data']['instance_score'])
    if(preds['data']['instance_score'][0] > 0.00342):
    # if(preds['data']['is_outlier'][0]==1):
        print("defect")
        ser.write('#'.encode())
    else:
        print("normal")
        ser.write('@'.encode())

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def putar(image):
    # image = cv2.imread('/media/ilham/Data 3/ubuntu/Project_an/dataset/foto/train/defect/defect_u/869.png')
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    if(angle != 0):
        rotated = rotate(image, angle, (38,33,31))
    else :
        rotated = rotate(image,0,(38,33,31))
    # rotated = imutils.rotate(image,angle)
    return rotated
if __name__ == "__main__":
    main_function()