#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
(train_X,train_Y), (test_X,test_Y) = mnist.load_data()


# In[2]:


print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)


# In[3]:


import numpy as np

classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# In[4]:


train_X
train_Y
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X.shape, test_X.shape


# In[5]:


from keras.utils import normalize
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = normalize(train_X, axis=1)
test_X = normalize(test_X, axis=1)


# In[6]:


from keras.utils import to_categorical

train_Y_argmax = to_categorical(train_Y)
test_Y_argmax = to_categorical(test_Y)

print('Original label:', train_Y[0])
print('After conversion to argmax:', train_Y_argmax[0])


# In[7]:


from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_argmax, test_size=0.2, random_state=13)
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


# In[8]:


from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
batch_size = 200
epochs = 10
learn_rate = 0.01
k_size = 3
feature_map = 32
pooling_size = 10


# In[9]:


model = Sequential()
model.add(Conv2D(feature_map, kernel_size=(k_size, k_size),activation='relu',input_shape=(28,28,1),padding='same'))
model.add(MaxPooling2D((pooling_size, pooling_size),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))


# In[ ]:


from keras import losses
from keras import optimizers

model.compile(loss=losses.categorical_crossentropy, 
              optimizer=optimizers.Adam(learning_rate=learn_rate),
              metrics=['accuracy'])


train = model.fit(train_X, train_label, batch_size = batch_size, epochs=epochs, validation_data = (valid_X, valid_label))


# In[ ]:


test_eval = model.evaluate(test_X, test_Y_argmax)


# In[ ]:


import matplotlib.pyplot as plt

accuracy = train.history['accuracy']
val_accuracy = train.history['val_accuracy']
loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', color="g", label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', color="g", label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


batch_size = 200
epochs = 10
learn_rate = 0.01

model_mlp = Sequential()
model_mlp.add(Flatten())
model_mlp.add(Dense(128, activation='relu'))
model_mlp.add(Dropout(0.25))
model_mlp.add(Dense(64, activation='relu'))
model_mlp.add(Dropout(0.25))
model_mlp.add(Dense(10, activation='softmax'))


# In[ ]:


model_mlp.compile(loss=losses.categorical_crossentropy, 
              optimizer=optimizers.Adam(learning_rate=learn_rate),
              metrics=['accuracy'])


train_mlp = model_mlp.fit(train_X, train_label, batch_size = batch_size, epochs=epochs, validation_data = (valid_X, valid_label))


# In[ ]:


test_eval = model_mlp.evaluate(test_X, test_Y_argmax)


# In[ ]:


accuracy = train_mlp.history['accuracy']
val_accuracy = train_mlp.history['val_accuracy']
loss = train_mlp.history['loss']
val_loss = train_mlp.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', color="g", label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', color="g", label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




