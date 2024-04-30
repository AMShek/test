# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:40:27 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 11:15:42 2020

@author: Administrator
"""

import cv2
import numpy as np
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((5 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

#print(objp)

corners=np.array([[[689., 881.]],
          [[1021., 861.]],
          [[1345., 849.]],
          [[1653., 837.]],
          [[1961., 817.]],
          [[2261., 809.]],
          [[2569., 785.]],
          [[2869.,773.]],
          [[729., 1193.]],
          [[1049., 1181.]],
          [[1369., 1165.]],
          [[1669., 1141.]],
          [[1973., 1125.]],
          [[2277., 1113.]],
          [[2577., 1089.]],
          [[2869.,1081.]],        
          [[753.,1505.]],
          [[1069.,1485.]],
          [[1381.,1461.]],
          [[1689.,1445.]],
          [[1989.,1429.]],
          [[2289.,1413.]],
          [[2585.,1393.]],
          [[2881.,1377.]],       
          [[775.,1813.]],
          [[1097., 1789.]],
          [[1405., 1769.]],
          [[1705., 1753.]],
          [[2005., 1729.]],
          [[2301., 1709.]],
          [[2597., 1689.]],
          [[2889.,1677.]],     
          [[789.,2129.]],
          [[1113.,2093.]],
          [[1417.,2077.]],
          [[1729.,2045.]],
          [[2021.,2025.]],
          [[2317.,2005.]],
          [[2605.,1981.]],
          [[2857.,1957.]]],dtype=float)
#print(corners)
img_points.append(corners)

#test        
print("img_points:")
print(img_points)
print("*"*50)
print("len(img_points:)")
print(len(img_points))
print("*"*50)

print("obj_points:")
print(obj_points)
print("*"*50)
print("len(obj_points:)")
print(len(obj_points))
print("*"*50)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (8,5), None, None)
print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs ) # 平移向量  # 外参数

print("-----------------------------------------------------")
images = glob.glob("C:/Users/Administrator/Desktop/det/*.jpg")
   
img = cv2.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
print (newcameramtx)
print("------------------使用undistort函数-------------------")
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
cv2.imwrite('cali1.jpg', dst)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
cv2.imwrite('calibresult1.jpg', dst1)
print ("方法一:dst1的大小为:", dst1.shape)