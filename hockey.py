
import numpy as np
from scipy import signal
import cv2
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
import pickle
from tensorflow import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
batch_size = 64
epochs = 200
def KNN(u2,v2,max3,min3):
  c1=max3
  c2=min3
  clus1=[]
  clus2=[]
  for j in range(0,len(u2)):
    for k in range(0,len(u2[0])):
      if abs(c1-u2[j][k])>abs(u2[j][k]-c2) and u2[j][k]!=0:
        clus2.append(u2[j][k])
        c2=(c2*len(clus2)+u2[j][k])/(len(clus2)+1)
      elif u2[j][k]!=0:
        clus1.append(u2[j][k])
        c1=(c1*len(clus1)+u2[j][k])/(len(clus1)+1)
      if abs(c1-v2[j][k])>abs(v2[j][k]-c2) and v2[j][k]!=0:
        clus2.append(v2[j][k])
        c2=(c2*len(clus2)+v2[j][k])/(len(clus2)+1)
      elif v2[j][k]!=0:
        clus1.append(v2[j][k])
        c1=(c1*len(clus1)+v2[j][k])/(len(clus1)+1)
  return clus1

def intersection(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def find_cords(arr, dic):
    temp_xarray = []
    temp_yarray = []
    temp_xarray.append(dic[0])
    temp_xarray.append(dic[2])
    temp_yarray.append(dic[1])
    temp_yarray.append(dic[3])
    temp_xarray.append(arr[0])
    temp_xarray.append(arr[2])
    temp_yarray.append(arr[1])
    temp_yarray.append(arr[3])         
    x_min = min(temp_xarray)
    x_max = max(temp_xarray)
    y_min = min(temp_yarray)
    y_max = max(temp_yarray)

    return(x_min, y_min, x_max, y_max)
def compute(magnitude, angle):
    s = np.zeros(12)
    for idx in range(magnitude.shape[0]):
        for mag, ang in zip(magnitude[idx].reshape(-1), angle[idx].reshape(-1)):
            if ang == 0:
                s[0] += mag
            elif ang > 0 and ang <= 360: # The condition can be removed if always true
                s[int((ang-1)//30)] += mag
    return s
def bow(fd):
  neigh = KMeans(n_clusters=100)
  if len(fd)>100:
    neigh.fit(np.array(fd).reshape(-1,1))
    fd1={}
    kl=neigh.cluster_centers_
    for ij in range(0,100):
      X.append(kl[ij][0])
    return X
  else:
    P=[]
    return P
def hog1(feat1,mag3,ang3):
  fd= hog(feat1, orientations=9, pixels_per_cell=(8, 8),
          cells_per_block=(2, 2), multichannel=True)
  return bow(fd)

def featureExtraction(img3,img1):
    #lkwop(img3,img1,10000)
    
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    flow = optical_flow.calc(cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)

    u1 = flow[...,0]    
    v1 = flow[...,1]
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #u1,v1=optical_flow(img3,img1,15,0.01)
    u11=u1
    v11=v1
    max1=0
    min1=1000
    dict={}
    for j in range(0,len(u11)-1):
      for k in range(0,len(u11[0])-1):
        if True:
          u11[j][k]=abs(u1[j][k])
          if u11[j][k]>max1:
            max1=u11[j][k]
          kl=[0,j,k]
          if u11[j][k] not in dict:
            dict[u11[j][k]]=[]
          dict[u11[j][k]].append(kl)
          if u11[j][k]<min1:
            min1=u11[j][k]
    for j in range(0,len(v11)-1):
      for k in range(0,len(v11[0])-1):
        if True:
          v11[j][k]=abs(v1[j+1][k])
          if v11[j][k]>max1:
            max1=v11[j][k]
          kl=[1,j,k]
          if v11[j][k] not in dict:
            dict[v11[j][k]]=[]
          dict[v11[j][k]].append(kl)
          if v11[j][k]<min1:
            min1=v11[j][k]
    cluster=KNN(u11,v11,max1,min1)
    dict1={}
    for i in cluster:
      if i in dict:
        for j in dict[i]:
          kl=[j[1],j[2]]
          #print(kl)
          if str(kl) in dict1:
            dict1[str(kl)]=2
          else:
            dict1[str(kl)]=1
    cvb=0
    dict2=[]
    for i in dict1:
      if dict1[i]==2:
        kl=list(map(int,i[1:-1].split(",")))
        for ui in boxes:
          if int(kl[1])>int(ui[0]) and int(kl[1])<int(ui[2]) and int(kl[0])>int(ui[1]) and int(kl[0])<int(ui[3]):
            q21=0
            for kj in dict2:
              uy=kj==ui
              if False not in uy:
                q21=1
            if q21==0:
              dict2.append(ui)
    cvb=0
    we1=0
    frame2=frame
    for i in dict1:
      if dict1[i]>1:
        we1=we1+1
        kl=list(map(int,i[1:-1].split(",")))
        cvb=cvb+abs(u1[kl[0]][kl[1]])+abs(v1[kl[0]][kl[1]])
    ans=0
    if we1>0:
      ans=cvb/we1
    return ans,magnitude,angle

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
import cv2
import numpy as np
import math
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import os
directory = 'data'
df=[]
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        df.append(f)

#training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)
X1=[]
acc2=0
nocount=0
couy=0
y=[]
X2=[]
for video in df:
  print(video)
  cap = cv2.VideoCapture(video)
  #print(video[13])
  fps = cap.get(cv2.CAP_PROP_FPS)
  print(fps)
  success, frame1 = cap.read()
  #frame1=imutils.rotate(frame1,angle=270)
  count=0
  county=0
  final=[]
  X=[]
  acc=0
  mag1=[]
  couy=couy+1
  print(couy)
  while success:
      success, frame = cap.read()
      #frame=imutils.rotate(frame,angle=270)
      if success and couy>5:
          county=county+1
          results=model(frame)
          predictions = results.pred[0]
          boxes = predictions[:, :4]
          categories = predictions[:, 5]
          boxes1=[]
          for i in range(0,len(categories)):
            if categories[i]==0:
              boxes1.append(boxes[i])
          boxes=boxes1
          box_dict={}
          lp=[]
          lp1=[]
          if len(boxes)>=1:
            p=0
            lp.append([])
            lp[0].append(boxes[0])
            lp1.append(boxes[0])
            for u7 in range(1,len(boxes)):
              iop=0
              for p7 in range(0,len(lp)):
                inte=intersection(lp1[p7],boxes[u7])
                if inte>0:
                  lp[p7].append(boxes[u7])
                  lp1[p7]=find_cords(lp1[p7],boxes[u7])
                  iop=1
                  break
              if iop==0:
                lp.append([])
                lp[len(lp)-1].append(boxes[u7])
                lp1.append(boxes[u7])
          hotspot_list=[]
          for iu in range(0,len(lp)):
            if len(lp[iu])>=1:
              hotspot_list.append(lp1[iu])
          maxq=0
          feat2=[]
          s3=[]
          for q123 in range(0,len(hotspot_list)):
            img=frame
            X=[]
            R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
            img1 = 0.2989 * R + 0.5870 * G + 0.1140 * B
            img2 = frame1
            R, G, B = img2[:,:,0], img2[:,:,1], img2[:,:,2]
            img3 = 0.2989 * R + 0.5870 * G + 0.1140 * B
            img4=img1
            img5=img3
            cv2.imwrite('hockey'+str(couy)+'.png',img1)
            img1=img1[int(hotspot_list[q123][1]):int(hotspot_list[q123][3]),int(hotspot_list[q123][0]):int(hotspot_list[q123][2])]
            img3=img3[int(hotspot_list[q123][1]):int(hotspot_list[q123][3]),int(hotspot_list[q123][0]):int(hotspot_list[q123][2])]
            feat1=frame[int(hotspot_list[q123][1]):int(hotspot_list[q123][3]),int(hotspot_list[q123][0]):int(hotspot_list[q123][2])]
            feat12=frame1[int(hotspot_list[q123][1]):int(hotspot_list[q123][3]),int(hotspot_list[q123][0]):int(hotspot_list[q123][2])]
            cv2.imwrite('hockey1'+str(couy)+'.png',img1)
            if count>0:
              count=0
              frame1=frame
            else:
              ans,magnitude,angle=featureExtraction(feat12,feat1)
              maxq=max(maxq,ans)
              if maxq==ans:
                feat2=feat1
                mag1=magnitude
                ang1=angle
          if maxq>0 and len(feat2)>10 and len(feat2[0])>10:
            X=hog1(feat2,mag1,ang1)
          else:
            X=[0 for i in range(0,100)]
          if len(X)>0:
              X1.append(X)
              if len(X1)>15:
                NH=[]
                for ui in range(0,16):
                          NH.append(X1[ui])
                X2.append(NH)
                if video[5]=='f':
                          y.append(1)
                else:
                          y.append(0)
                X1.pop(0)
  X1=[]
  X=[]          

Xmain=np.asarray(X2).astype('float32')
Ymain=np.asarray(y)
Ymain=Ymain.reshape(-1,1)
Ymain=to_categorical(Ymain)
Xmain=Xmain.reshape(-1,16,100,1)
batch_size = 20

#filename = 'finalized_model.sav'
#pickle.dump(clf, open(filename, 'wb'))
train_X,valid_X,train_label,valid_label = train_test_split(Xmain, Ymain, test_size=0.25, random_state=25)
#Ymain=to_categorical(Ymain)
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(16,100,1,),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.01))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.01))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.01))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.01))                  
fashion_model.add(Dense(2, activation='softmax'))
fashion_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.002),metrics=['accuracy'])
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
#filename = 'finalized_model1.sav'
#pickle.dump(clf, open(filename, 'wb'))
print(Xmain.shape)
print(train_label.shape)
print(valid_label)
print(acc2)
print(nocount)


