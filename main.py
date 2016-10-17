import numpy as np
import  cv2
import time
from operator import itemgetter
from load_face_models import *
from face_detection import *


caffe_root = '/Users/lixile/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

net_12c, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal=load_face_models(loadNet=True)

nets=(net_12c, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal)

img_path = '/Users/lixile/Documents/Research/FDDB/originalPics/2002/07/19/big/img_352.jpg'
#img_path = '/Users/lixile/Downloads/WIDER_train/images/12--Group/12_Group_Group_12_Group_Group_12_51.jpg'
img = cv2.imread(img_path)

small_face_size = 48
stride =5

img_data = np.array(img, dtype=np.float32)
img_data -= np.array((104, 117, 123))

rectangles = detect_faces_net(nets, img_data, small_face_size, stride, True, 2, 0.05)
for rectangle in rectangles:
    cv2.rectangle(img, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)

