#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[2]:


os.chdir("F:\\datasets\\AV\\train")
train = pd.read_csv('train.csv')


# In[3]:


train.head(5)


# In[4]:


train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('images/'+train['image'][i], target_size=(64, 64, 3), grayscale=False)
    #img = image.load_img('train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)


# In[5]:


#as it is a multy class problem , we should make a ht_encode to it
y=train['category'].values
y = to_categorical(y)


# In[6]:


#creating a validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=53, test_size=0.3)


# In[7]:


#creating the core model
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3),activation='relu',input_shape=(64, 64, 3)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
#model.add(Dense(6, activation = 'sigmoid'))


# In[8]:


model.summary()


# In[9]:


model.compile(optimizer='Adam' ,loss='categorical_crossentropy',metrics=['accuracy'])


# In[10]:


#training model
model.fit(X_train, y_train, epochs=125, validation_data=(X_test, y_test))


# In[11]:


#importing the test file and image files to predict
test = pd.read_csv('test.csv')


# In[12]:


test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('images/'+test['image'][i], target_size=(64, 64, 3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)


# In[13]:


#making predictions
prediction =model.predict_classes(test)


# In[14]:


print(prediction)


# In[15]:


# creating submission file
sample = pd.read_csv('test.csv')
sample['category'] = prediction
sample.to_csv('sample_prediction7.csv', header=True, index=False)


# In[16]:


sample


# In[ ]:




