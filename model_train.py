from numpy import *

import os
import numpy as np
import pandas as pd
import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D
BATCH_SIZE = 50
NUM_CLASSES = 2
NUM_EPOCHS = 10
MASK_SIZE = 40
index_low = int((100-MASK_SIZE)/2)
index_high = int((100+MASK_SIZE)/2)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

train_data_path='./train_val'
train_label_path='./'
test_path='./test'

import os
train_npz = np.load(train_data_path+'/candidate'+str(1)+'.npz')
data_train=[train_npz for p in range(465)]
data_trainvoxel=[train_npz['voxel'] for p in range(465)]
data_trainseg=[train_npz['seg'] for p in range(465)]
j=0
i=0
while (i<584) :
    if(os.path.exists(train_data_path+'/candidate'+str(i)+'.npz')):
        train_npz = np.load(train_data_path+'/candidate'+str(i)+'.npz')
        data_train[j]=train_npz
        data_trainvoxel[j]=data_train[j]['voxel']
        data_trainseg[j]=data_train[j]['seg']
        j+=1
    i+=1
data_train=data_trainvoxel*(np.array(data_trainseg).astype(int))
np.shape(data_train)

label_train1=pd.read_csv(train_label_path+'/train_val.csv')
label_train=[1 for p in range(465)]
for z in range(265):
    label_train[z]=label_train1['lable'][z]
label_train1=label_train


import os
test_npz = np.load(test_path+'/candidate'+str(11)+'.npz')
data_test=[test_npz for p in range(117)]
data_testvoxel=[test_npz['voxel'] for p in range(117)]
data_testseg=[test_npz['seg'] for p in range(117)]
label_testname=[' ' for p in range(117)]
j=0
i=0
while (i<583) :
    if(os.path.exists(test_path+'/candidate'+str(i)+'.npz')):
        test_npz = np.load(test_path+'/candidate'+str(i)+'.npz')
        data_test[j]=test_npz
        data_testvoxel[j]=data_test[j]['voxel']
        data_testseg[j]=data_test[j]['seg']
        label_testname[j]='candidate'+str(i)
        j+=1
    i+=1
data_test=data_testvoxel*(np.array(data_testseg).astype(int))
np.shape(data_test)


data_train = data_train.astype('float32') / 255.
data_test = data_test.astype('float32') / 255.
data_train2 = data_train[:,index_low:index_high,index_low:index_high,index_low:index_high]
data_test2 = data_test[:,index_low:index_high,index_low:index_high,index_low:index_high]
label_train = keras.utils.to_categorical(label_train, NUM_CLASSES)
data_train2 = np.reshape(data_train2,(465,MASK_SIZE,MASK_SIZE,MASK_SIZE,1))
data_test2 = np.reshape(data_test2,(117,MASK_SIZE,MASK_SIZE,MASK_SIZE,1))

np.shape(data_train2)

model = Sequential()
model.add(Conv3D(10,kernel_size=3 , input_shape=(MASK_SIZE,MASK_SIZE,MASK_SIZE,1), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2,1)))
model.add(Conv3D(index_low,kernel_size=3, activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2,1)))


model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(data_train2, label_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

model.save_weights('./weight.h5')

predict_data0=model.predict(data_test2)
predict_data=predict_data0[:,1]



dataframe = pd.DataFrame({'Id':label_testname,'Predicted':predict_data})
dataframe.to_csv(train_label_path+'/submission.csv',index=False,sep=',')



from sklearn.metrics import roc_auc_score
c=model.predict(data_train2)
d=[0 for p in range(465)]
for z in range(465):
    if label_train1[z]==0:
        d[z]=c[z][0]
    else:
        d[z]=c[z][1]
roc_auc_score(label_train1,d)



