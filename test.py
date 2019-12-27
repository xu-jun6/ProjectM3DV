from numpy import *

import os
import numpy as np
import pandas as pd
import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D

NUM_CLASSES = 2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

test_path='./test'

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

data_test = data_test.astype('float32') / 255.
data_test2 = data_test[:,30:70,30:70,30:70]
data_test2 = np.reshape(data_test2,(117,40,40,40,1))

model = Sequential()
model.add(Conv3D(10,kernel_size=3 , input_shape=(40,40,40,1), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2,1)))
model.add(Conv3D(30,kernel_size=3, activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2,1)))


model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(data_train2, label_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
model.load_weights('./weight.h5')

predict_data0=model.predict(data_test2)
predict_data=predict_data0[:,1]



dataframe = pd.DataFrame({'Id':label_testname,'Predicted':predict_data})
dataframe.to_csv('./'+'/submission.csv',index=False,sep=',')



#from sklearn.metrics import roc_auc_score
#c=model.predict(data_train2)
#d=[0 for p in range(465)]
#for z in range(465):
#    if label_train1[z]==0:
#        d[z]=c[z][0]
#    else:
#        d[z]=c[z][1]
#roc_auc_score(label_train1,d)



