#!/usr/bin/env python
# coding: utf-8


from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from progressbar import *
from tqdm import tqdm
import pandas as pd



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
try:
    os.mkdir('img2train')
except:
    print('img2train folder already exist')

try:
    os.mkdir('img2test')
except:
    print('img2test folder already exist')
    
    


train_csv = {'file':[],'label':[]}
t = tqdm(train_data.index,desc="cropping train")
for i in t:
    img = cv2.imread('images/'+train_data['filename'][i])
#     img = img[:,:,::-1]
    x_min = train_data['xmin'][i]
    x_max = train_data['xmax'][i]
    y_min = train_data['ymin'][i]
    y_max = train_data['ymax'][i]
    img2 = img[y_min:y_max,x_min:x_max,:]
    image = cv2.resize(img2,(80,80))
    name = 'train_%04d.jpg'%i
    train_csv['file'].append(name)
    train_csv['label'].append(train_data['label'][i])
    cv2.imwrite('img2train/'+name,image)
train_df = pd.DataFrame(train_csv)
train_df.to_csv('train_crop.csv',index=False)


test_csv = {'file':[],'label':[]}
t = tqdm(test_data.index,desc='cropping test')
for i in t:
    img = cv2.imread('images/'+test_data['filename'][i])
#     img = img[:,:,::-1]
    x_min = test_data['xmin'][i]
    x_max = test_data['xmax'][i]
    y_min = test_data['ymin'][i]
    y_max = test_data['ymax'][i]
    img2 = img[y_min:y_max,x_min:x_max,:]
    image = cv2.resize(img2,(80,80))
    name = 'test_%04d.jpg'%i
    test_csv['file'].append(name)
    test_csv['label'].append(test_data['label'][i])
    cv2.imwrite('img2test/'+name,image)
test_df = pd.DataFrame(test_csv)
test_df.to_csv('test_crop.csv',index=False)

print('image process finish')
