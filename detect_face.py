# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:47:26 2018

@author: chen
"""

import cv2 
import numpy as np

# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值

face_cascade = cv2.CascadeClassifier(r'./data/haarcascade_frontalface_default.xml')

# 待检测的图片路径

imagepath = r'./image/test.jpg'

# 读取图片

image = cv2.imread(imagepath)

'''
#显示图片
cv2.imshow("image",image)
cv2.waitKey(0)
#cv2.destroyAllWindows()
'''

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# 探测图片中的人脸

faceRects  = face_cascade.detectMultiScale(

    gray,

    scaleFactor = 1.15,

    minNeighbors = 5,

    minSize = (5,5),

)

if len(faceRects) > 0:  # 大于0则检测到人脸  
    for faceRect in faceRects:  # 单独截出每一张人脸  
        x, y, w, h = faceRect  
        faces = image[x-10 : x + w +20  , y -10 : y + h - 20]
        #cv2.rectangle(image, (x, y), (x + w , y + h ), (0,255,0)) #框出人脸
        cv2.imshow("face",faces)
        cv2.waitKey(0)
# 写入图像
cv2.imwrite('./image/face.jpg',faces)  



