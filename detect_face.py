# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:47:26 2018

@author: chen
"""

import cv2 
import numpy as np
import os

'''
#显示图片
cv2.imshow("image",image)
cv2.waitKey(0)
#cv2.destroyAllWindows()
'''

#重命名函数，参数文件夹路径，对需要进行预处理的照片进行重命名
def rename(path):
    i=0
    filelist = os.listdir(path)

    for files in filelist:  
        i=i+1
        Olddir = os.path.join(path,files)
        if os.path.isdir(Olddir): #如果是文件夹则跳过
            continue
        #filename = os.path.spiltext(files)[0] #文件名
        filetype = os.path.splitext(files)[1] #文件扩展名
        Newdir = os.path.join(path,str(i)+filetype)
        os.rename(Olddir,Newdir)



#预处理函数，对test文件夹中的测试照片进行预处理
#将处理之后的照片放入faces文件夹中，每个建立一个文件夹，对每张脸进行编号
        
def protreat():    
    # 获取训练好的人脸的参数数据
    face_cascade = cv2.CascadeClassifier(r'./data/haarcascade_frontalface_default.xml')
    # 待检测的图片路径
    imagepath = r'./image/test'
    # 存入识别之后的人脸路径
    facespath = r'./image/faces/'
    
    rename(imagepath)  #待检测的图片按进行重命名1.jpg，2.jpg。。。。
    #将每张图片进行处理
    i = 0
    for photo in os.listdir(imagepath):
        i = i + 1
        # 读取图片
        image = cv2.imread(imagepath+'/'+str(i)+'.jpg')    
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        #在faces目录下建立相应文件夹
        os.mkdir(os.path.join(facespath,str(i)))
        
        # 探测图片中的人脸
        faceRects  = face_cascade.detectMultiScale(
              gray,scaleFactor = 1.15,minNeighbors = 5,minSize = (5,5),
        )

        if len(faceRects) > 0:  # 大于0则检测到人脸  
            for faceRect in faceRects:  # 单独截出每一张人脸  
                x, y, w, h = faceRect  
                faces = image[x-10 : x + w +20  , y -10 : y + h - 20]  #截取人脸
                #进行放缩，统一成64*64
                faces = cv2.resize(faces,(64,64),interpolation=cv2.INTER_CUBIC)
                #cv2.rectangle(image, (x, y), (x + w , y + h ), (0,255,0)) #框出人脸
                ##展示截取的脸
                #cv2.imshow("face",faces)
                #cv2.waitKey(0)
                
                #cv2.imwrite(facespath+'/'+str(i)+'/'+str(i)+'_',faces)  # 写入图像

if __name__ == "__main__":
    protreat()

