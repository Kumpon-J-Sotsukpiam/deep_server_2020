import os
import shutil
import sys
import numpy as np
import cv2
import time
import math
from shutil import copyfile,rmtree

def cloneDir(rootDataset,toPath,temp_name,nameList):
    mkdir(toPath,temp_name=temp_name,sub2_path=nameList)

def precessData(rootDataset,nameList,per_valid):
    y = []
    for x in nameList:
        rootX = os.path.join(rootDataset,x)
        rootX_len = len(os.listdir(rootX))
        rootX_train_size = math.floor(rootX_len - (rootX_len*per_valid))
        rootX_valid_size = rootX_len - rootX_train_size
        y.append([x,rootX_len,rootX_train_size,rootX_valid_size])
    return y
def normalization(dataProcessed):
    # [x1,x2,x3,x4] => label,sum,train_size,valid_size
    sumArray = []
    for x in dataProcessed:
        sumArray.append(x[2])
    minOfData =  np.min(sumArray)
    maxOfData =  np.max(sumArray)
    averageOfData =  np.average(sumArray)
    print(minOfData,maxOfData,averageOfData)
    pass

def copyDataProcess(dataProcessed,rootDataset,toPath):
    train_temp_path = os.path.join(toPath,'data')
    eval_temp_path = os.path.join(toPath,'eval')
    for x in dataProcessed:
        x_path = os.path.join(rootDataset,x[0])
        x_train_temp_path = os.path.join(train_temp_path,x[0])
        x_eveal_temp_path = os.path.join(eval_temp_path,x[0])
        for i,x2 in enumerate(os.listdir(x_path),start=1):
            copyX2 = os.path.join(x_path,x2)
            copyX2_train = os.path.join(x_train_temp_path,x2)
            copyX2_test = os.path.join(x_eveal_temp_path,x2)
            if(i <= x[2]):
                copyfile(copyX2,copyX2_train)
            elif(i <= (x[2]+x[3])):
                copyfile(copyX2,copyX2_test)

def mkdir(temp_path="./temp",temp_name="temp_test",sub_path=["data","eval"],sub2_path=[]):
    make_path = []
    make_path.append(os.path.join(temp_path,temp_name))
    for name in sub_path:
        make_path.append(os.path.join(make_path[0],name))
    i=1
    while i<=len(sub_path):
        for name in sub2_path:
            make_path.append(os.path.join(make_path[i],name))
        i = i+1
    for mp in make_path:
        os.mkdir(mp)
    return make_path

def delTree(path):
    return shutil.rmtree(path)
